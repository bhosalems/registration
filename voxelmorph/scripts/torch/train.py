#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import time
import numpy as np
import torch

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'

import sys
sys.path.append("/home/csgrad/mbhosale/Image_registration/registration/voxelmorph")
import voxelmorph as vxm # nopep8
from voxelmorph.py.utils import *
from voxelmorph.generators import *
from voxelmorph.torch.networks import *
from voxelmorph.torch.losses import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from pathlib import Path

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--seg-list', help='line-seperated list of segmentation files')
parser.add_argument('--dataset', required=True)
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

bidir = args.bidir

logdir = "/".join(str(Path(__file__)).split("/")[:-3])
logdir = os.path.join(logdir, 'log/tb/' f'{datetime.now().strftime("%m%d%H%M")}')
if not os.path.exists(logdir):
    os.makedirs(logdir)
logfile = os.path.join(logdir, 'log.txt')
tb = SummaryWriter(comment = logfile)
candi_labels = [2,3,4,7,8,10,11,12,13,14,15,16,17,18,24,28,\
            41,42,43,46,47,49,50,51,52,53,54,60]

# load and prepare training data
train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)

# load corresponding segmentation files
seg_files = vxm.py.utils.read_file_list(args.seg_list, prefix=args.img_prefix, suffix=args.img_prefix)

assert len(train_files) > 0, 'Could not find any training data.'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    generator = vxm.generators.scan_to_atlas(train_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator
    generator = vxm.generators.scan_to_scan(
        train_files, segs=seg_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis, dataset=args.dataset)

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.torch.networks.VxmDense.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.torch.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
losses += [Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

epoch_train_dice = AverageMeter()
globalidx = 0
handlers = [logging.StreamHandler()]
handlers.append(logging.FileHandler(
        logfile, mode='a'))
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,     
    )

# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if epoch % 20 == 0:
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(generator)
        source_seg = inputs[-1]
        target_seg = y_true[-1]
        inputs = inputs[:-1]
        y_true = y_true[:-1]
        
        if args.dataset == "CANDI":
            seg = np.zeros_like(source_seg)
            for n,label in enumerate(candi_labels):
                newlabel = n+1
                seg[source_seg==label]=newlabel
            source_seg = torch.from_numpy(seg).to(device).float().permute(0, 4, 1, 2, 3)
            seg = np.zeros_like(target_seg)
            for n,label in enumerate(candi_labels):
                newlabel = n+1
                seg[target_seg==label]=newlabel
            target_seg = torch.from_numpy(seg).to(device).float().permute(0, 4, 1, 2, 3)
        
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
        
        print(source_seg.unique())
        print(target_seg.unique())
        
        source_seg = torch.squeeze(source_seg, 1).float().cuda()
        source_seg = torch.nn.functional.one_hot(source_seg.long(), num_classes=29).float().permute(0,4,1,2,3)
        target_seg = torch.squeeze(target_seg, 1).float().cuda()
        target_seg = torch.nn.functional.one_hot(target_seg.long(), num_classes=29).float().permute(0,4,1,2,3)
        
        # run inputs through the model to produce a warped image and flow field
        output = model(*inputs, moving_seg=source_seg, fixed_seg=target_seg)
        y_pred = output[:2]
        dice = output[2]  
        
        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())
        epoch_train_dice.update(dice)
        if tb is not None:
            tb.add_scalar("train/loss", loss.item(), globalidx)        
        
        logging.info("Epoch [{}/{}] steps[{}/{}] Dice score:{} Epoch loss:{}".format(epoch, args.epochs+1, step, args.steps_per_epoch, dice, loss.item()))
        globalidx+=1
        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # print epoch info
    epoch_info = '[Epoch %d/%d]' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
    if tb is not None:
        tb.add_scalar("train/dice", epoch_train_dice.avg, epoch)
        tb.add_scalar("train/total_loss", np.mean(epoch_total_loss), epoch)
        
            

# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))