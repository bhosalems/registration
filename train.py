import torch
import logging
from utils import *
import os
from losses import *
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from uncert.snapshot import CyclicCosAnnealingLR


class TrainModel():
    def __init__(self, model, train_dataloader, test_dataloader, args, n_class, tb=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.tb = tb
        self.args = args    
        self.n_class = n_class
        self.printfreq = 1
        self.cur_epoch = 0
        self.cur_idx = 0
        
        #
        if args.logfile:
            savepath = f'ckpts/{args.logfile}'
        else:
            savepath = f'ckpts/vm'
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        self.savepath = savepath
        #
    def trainIter(self, fix, moving, fixed_label, moving_label, fixed_nopad=None, seg_f=None):
        if not os.path.exists('seg_imgs'):
            os.mkdir('seg_imgs')
        seg_fname = "seg_imgs/e"+str(self.cur_epoch)+"idx"+str(self.cur_idx)
        if seg_f is not None:
            seg_fname = seg_fname + "_" + seg_f
        sim_loss, grad_loss, dice = self.model.forward(fix, moving, fixed_label, moving_label, fix_nopad=fixed_nopad, 
                                                       rtloss=True, eval=True, dice_labels=self.train_dataloader.dataset.dice_labels, 
                                                       seg_fname=seg_fname)
        sim_loss, grad_loss, dice = sim_loss.mean(), grad_loss.mean(), dice.mean()
        loss = float(self.args.weight[0])*sim_loss + float(self.args.weight[1])*grad_loss
        if self.global_idx%self.printfreq ==0:
            logging.info(f'simloss={sim_loss}, gradloss={grad_loss}, loss={loss}, dice={(dice*100):2f}')
        if self.tb is not None:
            self.tb.add_scalar("train/loss", loss.item(), self.global_idx)
            self.tb.add_scalar("train/grad_loss", grad_loss.item(), self.global_idx)
            self.tb.add_scalar("train/sim_loss", sim_loss.item(), self.global_idx)
        return loss, dice

    def data_extract(self, samples):
        if len(samples)==4:
            fixed, fixed_label, moving, moving_label = samples
            fixed_nopad = None
        else:
            fixed, fixed_label, fixed_nopad, moving, moving_label = samples
            
        # Mahesh : Q. Why we need to unsqeeze? >> Make depth/channel as second dimension for conv, i.e. our volume is gray sclae, so it's 1.
        moving = torch.unsqueeze(moving, 1).float().cuda()
        fixed = torch.unsqueeze(fixed, 1).float().cuda()
        # moving_label = torch.unsqueeze(moving_label, 1).float().cuda()
        # fixed_label = torch.unsqueeze(fixed_label, 1).float().cuda()
        fixed_label = fixed_label.float().cuda()
        fixed_label = torch.nn.functional.one_hot(fixed_label.long(), num_classes=self.n_class).float().permute(0,4,1,2,3)
        moving_label = moving_label.float().cuda()
        moving_label = torch.nn.functional.one_hot(moving_label.long(), num_classes=self.n_class).float().permute(0,4,1,2,3)
        
        if fixed_nopad is not None:
            # moving_label = fixed_nopad * moving_label
            fixed_nopad = fixed_nopad.float().cuda()[:, None]
            fixed_label = fixed_nopad * fixed_label
        
        # Mahesh : Q. Why do we need to permute here, Is it okay if we do not onehot code? >> To make the class/label dimension second dimension,
        # likely required by the loss.
        # fixed_label = torch.nn.functional.one_hot(fixed_label.long(), num_classes=self.n_class).float().permute(0, 4, 1, 2, 3).cuda()
        # moving_label = torch.nn.functional.one_hot(moving_label.long(), num_classes=self.n_class).float().permute(0, 4, 1, 2, 3).cuda()
        return fixed, fixed_label, moving, moving_label, fixed_nopad

    def test(self, epoch):
        tst_dice = AverageMeter()
        self.model.eval()
        logging.info("Evaluation started")
        idx = 0
        for _, samples in enumerate(self.test_dataloader):
            p = int(samples[-1])
            samples = samples[:-1]
            fixed, fixed_label, moving, moving_label, fixed_nopad = self.data_extract(samples)
            dice = self.model.forward(fixed, moving,  fixed_label, moving_label, fixed_nopad, rtloss=False, eval=True, 
                                      seg_fname='seg_imgs/vale'+str(epoch)+'idx'+str(idx), dice_labels=self.train_dataloader.dataset.dice_labels)
            dice = dice.mean()
            tst_dice.update(dice.item())
            idx+=1
        #epoch

        if self.tb is not None:
            self.tb.add_scalar("test/dice", tst_dice.avg, epoch)
        return tst_dice.avg
    
    def train_epoch(self, optimizer, scheduler, epoch):
        epoch_train_dice = AverageMeter()
        self.model.train()
        idx = 0
        for _, samples in enumerate(self.train_dataloader):
            p = int(samples[-1])
            samples = samples[:-1]
            fixed, fixed_label, moving, moving_label, fixed_nopad = self.data_extract(samples)
            self.global_idx += 1
            self.cur_idx = idx
            # seg_fname = self.train_dataloader.dataset.imgpath[self.train_dataloader.dataset.pairs[p][1]].split('/')[-4]
            seg_fname = self.train_dataloader.dataset.imgpath[p].split('/')[-4]
            logging.info(f'iteration={idx}/{len(self.train_dataloader)}')
            trloss, trdice = self.trainIter(fixed, moving, fixed_label, moving_label, fixed_nopad=fixed_nopad, seg_f=seg_fname)
            optimizer.zero_grad()
            trloss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_train_dice.update(trdice.item())
            idx+=1
        #epoch
        
        if self.tb is not None:
            self.tb.add_scalar("train/dice", epoch_train_dice.avg, epoch)
        
        #validate per epoch
        # import ipdb; ipdb.set_trace()
        dice = self.test(epoch)
        if dice>self.bestdice:
            self.bestdice = dice
            savename = os.path.join(self.savepath, f'{epoch}_{(dice*100):2f}.ckpt')
            torch.save(self.model.state_dict(), savename)
        logging.info(f'Epoch:{epoch}...TestDice:{(dice*100):2f}, Best{(self.bestdice*100):2f}')

    def run(self):#device=torch.device("cuda:0")
        if self.args.sgd:
            optimizer = SGD(self.model.parameters(), lr = self.args.lr)
        else:
            optimizer = Adam(self.model.parameters(),lr = self.args.lr)

        # scheduler = StepLR(opt, step_size=5, gamma=0.1)
        scheduler = None

        self.global_idx = 0
        self.bestdice = 0
        for epoch in range(self.args.epoch):
            self.train_epoch(optimizer, scheduler, epoch)
            self.cur_epoch = epoch
    
    def run_snapshot(self, cycles=20, ):
        epochs = self.args.epoch
        epochs_per_cycle = epochs// cycles
        global_epoch = 0
        self.global_idx = 0
        self.bestdice = 0

        for n_cycle in range(cycles):
            if self.args.sgd:
                optimizer = SGD(self.model.parameters(), lr = 0.001)#0.01 lossNan
            else:
                optimizer = Adam(self.model.parameters(), lr = self.args.lr)#1e-4
            
            scheduler = CyclicCosAnnealingLR(optimizer, epochs_per_cycle)
            for epoch in range(epochs_per_cycle):
                global_epoch+=1
                self.train_epoch(optimizer, scheduler, global_epoch)

            savename = os.path.join(self.savepath, f'cycle{n_cycle}_{(self.bestdice*100):2f}.ckpt')
            torch.save(self.model.state_dict(), savename)


class TrainUncertModel(TrainModel):
    def __init__(self, tmodel, model, train_dataloader, test_dataloader, args, n_class, tb=None):
        self.tmodel = tmodel
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.tb = tb
        self.args = args
        self.n_class = n_class
        self.printfreq=50
        #
        if args.logfile:
            savepath = args.logfile
        else:
            savepath = './uncert'
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        self.savepath = savepath

    def trainIter(self, fix, moving, fixed_label, moving_label, ):
        gtflow = self.tmodel.forward(fix, moving,  fixed_label, moving_label, rtloss=False,eval=False).detach()
        loss, dice = self.model.forward(fix, moving,  fixed_label, moving_label, gtflow=gtflow, rtloss=True,eval=True)
        
        if self.global_idx%self.printfreq ==0:
            logging.info(f'loss={loss.item()}, dice={dice.item()}')
        if self.tb is not None:
            self.tb.add_scalar("train/loss", loss.item(), self.global_idx)
        return loss, dice