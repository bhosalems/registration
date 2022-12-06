import numpy as np
import random
from collections import defaultdict
import itertools
import os
import argparse
import logging
from datetime import datetime
import torch
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
# import nibabel as nib
import dataset.candi as candi
import dataset.msd as msd
import dataset.ixi as ixi
import dataset.braTS as brats
import dataset.chaos as chaos
import dataset.learn2reg as learn2reg
from torch.optim.lr_scheduler import StepLR
from train import TrainModel
from models import RegNet
import math

CANDI_PATH = r'/data_local/mbhosale/CANDI_split'
MSD_PATH = r'/data_local/mbhosale/MSD'
IXI_PATH = r'/home/csgrad/mbhosale/Image_registration/TransMorph_Transformer_for_Medical_Image_Registration/IXI/IXI_data/'
BraTS_PATH = r'/home/csgrad/mbhosale/Image_registration/datasets/BraTS2018'
BraTS_save_PATH = r'/home/csgrad/mbhosale/Image_registration/datasets/BraTS2018/'
CHAOS_PATH = r'/home/csgrad/mbhosale/Datasets/CHAOS_preprocessed/'
L2R_DATAPATH = r"/home/csgrad/mbhosale/Datasets/learn2reg/AbdomenMRCT/"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', action='store_true',
                        help='Fine tuning the model')
    parser.add_argument('--pretrain', type=str, default=None, help='pth name of pre-trained model')
    parser.add_argument('--dataset', type=str, default='CANDI', help='CANDI, prostate, IXI')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weightdecay', type=float, default=0, help='Weightdecay')
    parser.add_argument('--epoch', type=int, default = 500, help = "Max Epoch")
    parser.add_argument('--bsize', type=int, default=2, help='Batch size') 
    parser.add_argument('--num_workers', type=int, default=4)
    # parser.add_argument('--savefrequency', type=int, default=25, help='savefrequency')
    # parser.add_argument('--testfrequency', type=int, default=1, help='testfrequency')
    parser.add_argument('--gpu', default='0, 1', type=str, help='GPU device ID (default: -1)')
    parser.add_argument('--log', default='./logs/', type=str)
    parser.add_argument('--weight', type=str, default='1, 2', help='LAMBDA, GAMMA')
    # parser.add_argument('--uncert', type=int, default=0)
    # parser.add_argument('--dual', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--droprate', type=float, default=0)
    parser.add_argument('--sgd', action='store_true')
    # parser.add_argument('--feat', action='store_true')
    parser.add_argument('--tr_modality', type=str, default='T1DUAL', help='train modality')
    parser.add_argument('--tst_modality', type=str, default='T1DUAL', help='test modality')
    parser.add_argument('--tr_phase', type=str, default='InPhase', help='train phase')
    parser.add_argument('--tst_phase', type=str, default='InPhase', help='test phase')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    downsample_rate = 16
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.weight = [float(i) for i in args.weight.split(',')]
    
    handlers = [logging.StreamHandler()]
    if args.log:
        args.log = os.path.join(args.log, args.dataset, datetime.now().strftime("%m_%d_%y_%H_%M"))
        if not os.path.isdir(args.log):
            os.makedirs(args.log)
    #load model
    #device = torch.device(0)
    gpu = [int(i) for i in range(torch.cuda.device_count())]
    if args.dataset=='CANDI':
        pad_size=[160, 160, 128]
        window_r = 7
        NUM_CLASS = 29
        train_dataloader, test_dataloader = candi.CANDI_dataloader(args, datapath=CANDI_PATH, size=pad_size)
    elif args.dataset == 'prostate' or args.dataset == 'hippocampus' or args.dataset == "liver":
        NUM_CLASS = 3
        if args.dataset == 'hippocampus': 
            window_r = 5
        elif args.dataset == 'liver':
            window_r = 15
        else:
            window_r = 9
        if args.dataset == 'hippocampus':
            pad_size = [48, 64, 48]
        elif args.dataset == 'prostate': 
            pad_size = [240, 240, 96]
        else:
            pad_size = [256, 256, 128]
        train_dataloader, test_dataloader, _ = msd.MSD_dataloader(args.dataset, args.bsize, args.num_workers, pad_size, datapath=MSD_PATH)
    elif args.dataset == 'IXI':
        train_dataloader, test_dataloader = ixi.IXI_dataloader(datapath=IXI_PATH, batch_size=args.bsize, num_workers=4)
        pad_size = [160, 192, 224]
        window_r = 7
        NUM_CLASS = 46
    elif args.dataset == 'BraTS':
        pad_size = [240, 240, 155]
        train_dataloader, test_dataloader = brats.braTS_dataloader(root_path=BraTS_PATH, save_path = BraTS_save_PATH, 
                                                                   bsize=args.bsize, mod=args.modality, augment=False)
        pad_size = train_dataloader.dataset.size        
        window_r = 7
        # Mahesh : Should the mumber of classes be one more than total number of classes? As required for some of the loss functions etc. >> No Need, 
        # we are not using any other loss function such as cross entropy loss which takes in number of classes as an arguemnt.
        NUM_CLASS = 4
    elif args.dataset == 'CHAOS':
        pad_size = [256, 256, 50] # for T1DUAL [256, 256, 50] # TODO Why is it 320 for T2spir ? 
        tr_path = CHAOS_PATH + r"CHAOS_Train_Sets/Train_Sets/MR"
        tst_path = CHAOS_PATH + r"CHAOS_Train_Sets/Train_Sets/MR" # we are choosing train dataset as test because we dont have ground truth in the test
        # TODO But we can change the test modality, but for now we have kept it same 
        train_dataloader, test_dataloader = chaos.Chaos_dataloader(root_path=CHAOS_PATH,  tr_path=tr_path, tst_path=tst_path, 
                                                         bsize=1, tr_modality=args.tr_modality, tr_phase=args.tr_phase, tst_modality=args.tst_modality, 
                                                         tst_phase=args.tst_phase, size=pad_size, data_split=False, n_fix=1, tr_num_samples=0, 
                                                         tst_num_samples=10)
        if pad_size[-1]%downsample_rate != 0:               
            orig_size = pad_size
            c_dim = orig_size[-1]
            pad_size[-1] += abs(c_dim - (math.ceil(c_dim/downsample_rate)*downsample_rate))
        window_r = 11
        NUM_CLASS = 5
    elif args.dataset == "learn2reg":
        pad_size = [192, 160, 192]
        train_dataloader, test_dataloader = learn2reg.l2r_dataloader(datapath=L2R_DATAPATH, 
                                                                     size=pad_size, mod="MR", bsize=1, num_workers=1)
        window_r = 7
        NUM_CLASS = 5
         
    ##BUILD MODEL##
    model = RegNet(pad_size, winsize=window_r, dim=3, n_class=NUM_CLASS).cuda()
    
    if len(gpu)>1:
        model = torch.nn.DataParallel(model, device_ids=gpu)
    #model = nn.DataParallel(model, device_ids=gpu).to(device)
    # import ipdb; ipdb.set_trace()
    
    # if args.sgd:
    #     opt = SGD(model.parameters(), lr = args.lr)
    # else:
    #     opt = Adam(model.parameters(),lr = args.lr)

    # # scheduler = StepLR(opt, step_size=5, gamma=0.1)
    # scheduler = None
    # import ipdb; ipdb.set_trace()
    
    if args.debug:
        logfile = os.path.join(args.log, 'logs_wnidow_{window_r}.txt')
    else:
        logfile = os.path.join(args.logfile, f'{datetime.now().strftime("%m%d%H%M")}.txt')
    handlers.append(logging.FileHandler(
        logfile, mode='a'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,     
    )
    logging.info(args)
    device = torch.device("cuda")
    logging.info(f'Device: {device}')

    logging.info(f"DEVICE COUNT {torch.cuda.device_count()}")
    logging.info(f'GPU: {args.gpu}')
    
    if not args.debug:
        writer_comment = f'{args.logfile}'#'_'.join(['vm','un'+str(args.uncert), str(args.weight), args.logfile]) 
        tb = SummaryWriter(comment = writer_comment)
    else:
        writer_comment = './logs/tb'#'_'.join(['vm','un'+str(args.uncert), str(args.weight), args.logfile]) 
        tb = SummaryWriter(comment = writer_comment)

    train = TrainModel(model, train_dataloader, test_dataloader, args, NUM_CLASS, tb=tb)
    train.run()

    if tb is not None:
        tb.close()