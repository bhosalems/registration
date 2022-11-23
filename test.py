import torch
import logging
from utils import *
import os
from losses import *
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from uncert.snapshot import CyclicCosAnnealingLR
from datetime import datetime
import dataset.candi as candi
import dataset.msd as msd
import argparse
from models import RegNet

CANDI_PATH = r'/data_local/mbhosale/CANDI_split'
MSD_PATH = r'/data_local/mbhosale/MSD'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='prostate', help='CANDI, prostate, hippocampus, liver')
    parser.add_argument('--test_dataset', type=str, default='CANDI', help='CANDI, prostate, hippocampus, liver')
    parser.add_argument('--log', default='./logs/', type=str)
    parser.add_argument('--bsize', default=1, type=int)
    parser.add_argument('--checkpoint')
    return parser.parse_args()

class TestModel():
    def __init__(self, ckpt_file, tr_padsize, win_size, test_dataloader, n_class):
        self.test_dataloader = test_dataloader
        self.n_class = n_class
        model = RegNet(tr_padsize, winsize=win_size, dim=3, n_class=n_class).cuda()
        d = torch.load(ckpt_file)
        # Skit loading the spatial transformer, as the train and test datasets are different.
        d = {k: v for k, v in d.items() if k !='spatial_transformer_network.meshgrid'}
        model.load_state_dict(d, strict=False)
        model.eval()
        self.model = model

    def data_extract(self, samples):
        if len(samples)==4:
            fixed, fixed_label, moving, moving_label = samples
            fixed_nopad = None
        else:
            fixed, fixed_label, fixed_nopad, moving, moving_label = samples
            
        moving = torch.unsqueeze(moving, 1).float().cuda()
        fixed = torch.unsqueeze(fixed, 1).float().cuda()
        fixed_label = fixed_label.float().cuda()
        fixed_label = torch.nn.functional.one_hot(fixed_label.long(), num_classes=self.n_class).float().permute(0,4,1,2,3)
        moving_label = moving_label.float().cuda()
        moving_label = torch.nn.functional.one_hot(moving_label.long(), num_classes=self.n_class).float().permute(0,4,1,2,3)
        
        if fixed_nopad is not None:
            fixed_nopad = fixed_nopad.float().cuda()[:, None]
            fixed_label = fixed_nopad * fixed_label
        return fixed, fixed_label, moving, moving_label, fixed_nopad
    
    def test(self):
        tst_dice = AverageMeter()
        self.model.eval()
        logging.info(" started")
        idx = 0
        for _, samples in enumerate(self.test_dataloader):
            p = int(samples[-2])
            seg_fname = samples[-1]
            samples = samples[:-2]
            fixed, fixed_label, moving, moving_label, fixed_nopad = self.data_extract(samples)
            dice = self.model.forward(fixed, moving,  fixed_label, moving_label, fixed_nopad, rtloss=False, eval=True, 
                                      seg_fname=seg_fname, dice_labels=self.test_dataloader.dataset.dice_labels)
            dice = dice.mean()
            tst_dice.update(dice.item())
            logging.info(f'iteration={idx}/{len(self.test_dataloader)}')
            logging.info(f'{seg_fname}"test dice="{dice.item()}')
            idx+=1
        logging.info(f'Average test dice= {tst_dice.avg}')

if __name__ == "__main__":
    args = get_args()
    logfile = os.path.join(args.log, "cross_domain_test.txt")
    handlers = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(
        logfile, mode='a'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,     
    )
    logging.info(args)
    
    # Set things required for the testing
    if args.test_dataset == "prostate" or args.test_dataset == "liver" or args.test_dataset == "hippocampus":
        test_datapath = MSD_PATH
        if args.test_dataset == "hippocampus":
            tst_pad_size = [48, 64, 48]
        elif args.test_dataset == "liver":
            tst_pad_size = [256, 256, 128]
        else:
            tst_pad_size = [240, 240, 96]
        testseg_dataloader, _ = msd.MSD_test_dataloader(args.test_dataset, args.bsize, tst_pad_size, datapath=test_datapath)
    elif args.test_dataset == "CANDI":
        test_datapath = CANDI_PATH
        tst_pad_size=[160, 160, 128]
        testseg_dataloader, _ = candi.CANDI_test_dataloader(args.test_dataset, args.bsize, tst_pad_size, datapath=test_datapath)
        
    # Set things required for the training
    if args.train_dataset=='CANDI':
        tr_pad_size=[160, 160, 128]
        window_r = 7
        NUM_CLASS = 29
    elif args.train_dataset == 'prostate' or args.train_dataset == 'hippocampus' or args.train_dataset == "liver":
        NUM_CLASS = 3
        if args.train_dataset == 'hippocampus':
            tr_pad_size = [48, 64, 48]
            window_r = 5
        elif args.train_dataset == 'prostate': 
            tr_pad_size = [240, 240, 96]
            window_r = 9
        else:
            tr_pad_size = [256, 256, 128]
            window_r = 15
    
    test = TestModel(args.checkpoint, tst_pad_size, window_r, testseg_dataloader, NUM_CLASS)
    test.test()
    