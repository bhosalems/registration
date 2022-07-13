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
from torch.optim.lr_scheduler import StepLR
from train import TrainModel
from models import RegNet

CANDI_PATH = '~/data/CANDI_split'
MSD_PATH = r'C:\Users\mahes\Desktop\UB\Thesis\Img registration\registration\dataset\MSD'
IXI_PATH = r'/home/csgrad/mbhosale/Image_registration/TransMorph_Transformer_for_Medical_Image_Registration/IXI/IXI_data/'
BraTS_PATH = r'/home/csgrad/mbhosale/Image_registration/datasets/BraTS2018'
BraTS_save_PATH = r'/home/csgrad/mbhosale/Image_registration/datasets/BraTS2018/'

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
    # parser.add_argument('--savefrequency', type=int, default=1, help='savefrequency')
    # parser.add_argument('--testfrequency', type=int, default=1, help='testfrequency')
    parser.add_argument('--gpu', default='0, 1', type=str, help='GPU device ID (default: -1)')
    parser.add_argument('--logfile', default='', type=str)
    parser.add_argument('--weight', type=str, default='1,0.01', help='LAMBDA, GAMMA')
    # parser.add_argument('--uncert', type=int, default=0)
    # parser.add_argument('--dual', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--droprate', type=float, default=0)
    parser.add_argument('--sgd', action='store_true')
    # parser.add_argument('--feat', action='store_true')
    parser.add_argument('--modality', type=str, default='flair', help='Modality of the scan ' 
    'to be used, used only in case of BraTS dataset for now')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.weight = [float(i) for i in args.weight.split(',')]
    
    handlers = [logging.StreamHandler()]
    if args.debug:
        logfile = f'debug_071322'
    else:
        logfile = f'{args.logfile}-{datetime.now().strftime("%m%d%H%M")}'
    handlers.append(logging.FileHandler(
        f'./logs/{logfile}.txt', mode='a'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,     
    )
    logging.info(args)
    #load model
    #device = torch.device(0)
    device = torch.device("cuda")
    logging.info(f'Device: {device}')

    logging.info(f"DEVICE COUNT {torch.cuda.device_count()}")
    gpu = [int(i) for i in range(torch.cuda.device_count())]
    logging.info(f'GPU: {args.gpu}')

    if args.dataset=='CANDI':
        pad_size=[160, 160, 128]
        window_r = 7
        NUM_CLASS = 29
        train_dataloader, test_dataloader = candi.CANDI_dataloader(args, datapath=CANDI_PATH, size=pad_size)
    elif args.dataset == 'prostate' or args.dataset == 'hippocampus':
        train_dataloader, test_dataloader, _ = msd.MSD_dataloader(args.dataset, args.bsize, args.num_workers, datapath=MSD_PATH)
        NUM_CLASS = 3
        window_r = 5 if args.dataset == 'hippocampus' else 9
        pad_size = [48, 64, 48] if args.dataset=='hippocampus' else [240, 240, 96]
    elif args.dataset == 'IXI':
        train_dataloader, test_dataloader = ixi.IXI_dataloader(datapath=IXI_PATH, batch_size=args.bsize, num_workers=4)
        pad_size = [160, 192, 224]
        window_r = 7
        NUM_CLASS = 46
    elif args.dataset == 'BraTS':
        train_dataloader, test_dataloader = brats.braTS_dataloader(root_path=BraTS_PATH, save_path = BraTS_save_PATH, bsize=args.bsize, mod=args.modality)
        pad_size = [240, 240, 155]
        window_r = 7
        # Should the mumber of classes be one more than total number of classes? As required for some of the l
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
    
    if not args.debug:
        writer_comment = f'{args.logfile}'#'_'.join(['vm','un'+str(args.uncert), str(args.weight), args.logfile]) 
        tb = SummaryWriter(comment = writer_comment)
    else:
        print('Creating the tensorborad file here')
        writer_comment = './logs/tb'#'_'.join(['vm','un'+str(args.uncert), str(args.weight), args.logfile]) 
        tb = SummaryWriter(comment = writer_comment)

    train = TrainModel(model, train_dataloader, test_dataloader, args, NUM_CLASS, tb=tb)
    train.run()

    if tb is not None:
        tb.close()

    
    