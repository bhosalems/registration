import os, utils, glob, losses
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import TransMorph_affine
import torch
from torchvision import transforms
from torch import optim
import matplotlib.pyplot as plt
from TransMorph_affine import CONFIGS as CONFIGS
from natsort import natsorted
import math
import sys
sys.path.insert(0, "/home/csgrad/mbhosale/Image_registration/registration")
import dataset.chaos as chaos
from torch.utils.tensorboard import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
        
def data_extract(samples, n_class):
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
        fixed_label = torch.nn.functional.one_hot(fixed_label.long(), num_classes=n_class).float().permute(0,4,1,2,3)
        moving_label = moving_label.float().cuda()
        moving_label = torch.nn.functional.one_hot(moving_label.long(), num_classes=n_class).float().permute(0,4,1,2,3)
        
        if fixed_nopad is not None:
            # moving_label = fixed_nopad * moving_label
            fixed_nopad = fixed_nopad.float().cuda()[:, None]
            fixed_label = fixed_nopad * fixed_label
        
        # Mahesh : Q. Why do we need to permute here, Is it okay if we do not onehot code? >> To make the class/label dimension second dimension,
        # likely required by the loss.
        # fixed_label = torch.nn.functional.one_hot(fixed_label.long(), num_classes=self.n_class).float().permute(0, 4, 1, 2, 3).cuda()
        # moving_label = torch.nn.functional.one_hot(moving_label.long(), num_classes=self.n_class).float().permute(0, 4, 1, 2, 3).cuda()
        return fixed, fixed_label, moving, moving_label, fixed_nopad

if __name__ == "__main__":
    '''
    GPU configuration
    ''' 
    tb = SummaryWriter(comment = './logs/tb')
    save_model = True
    GPU_iden = 6
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))

    batch_size = 1
    save_dir = 'TransMorph_affine_NCC'
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    lr = 0.0001
    epoch_start = 0
    max_epoch = 500
    cont_training = False
    config = CONFIGS['TransMorph-Affine']
    AffInfer = TransMorph_affine.ApplyAffine()
    AffInfer.cuda()
    model = TransMorph_affine.SwinAffine(config)
    device = torch.device("cuda:"+str(GPU_iden) if torch.cuda.is_available() else "cpu")
    model.cuda()
    if cont_training:
        epoch_start = 335
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['state_dict']
        model.load_state_dict(best_model)
    else:
        updated_lr = lr
    print('Current learning rate: {}'.format(updated_lr))
    CHAOS_PATH = r'/home/csgrad/mbhosale/Datasets/CHAOS_preprocessed/'
    downsample_rate = 16
    save_freq = 50
    pad_size = [256, 256, 50] # for T1DUAL [256, 256, 50] # TODO Why is it 320 for T2spir ? 
    tr_path = CHAOS_PATH + r"CHAOS_Train_Sets/Train_Sets/MR"
    tst_path = CHAOS_PATH + r"CHAOS_Train_Sets/Train_Sets/MR" # we are choosing train dataset as test because we dont have ground truth in the test
    # TODO But we can change the test modality, but for now we have kept it same 
    train_dataloader, test_dataloader = chaos.Chaos_dataloader(root_path=CHAOS_PATH,  tr_path=tr_path, tst_path=tst_path, 
                                                        bsize=1, tr_modality='T1DUAL', tr_phase='InPhase', tst_modality='T1DUAL', 
                                                        tst_phase='InPhase', size=pad_size, data_split=False, n_fix=1, tr_num_samples=0, 
                                                        tst_num_samples=10)
    if pad_size[-1]%downsample_rate != 0:
        orig_size = pad_size
        c_dim = orig_size[-1]
        pad_size[-1] += abs(c_dim - (math.ceil(c_dim/downsample_rate)*downsample_rate)) 

    optimizer = optim.AdamW(model.parameters(), lr=updated_lr, amsgrad=True)
    Sim_loss = losses.NCC_vxm()
    print('Training Starts')
    for epoch in range(epoch_start, max_epoch):
        '''
        Training
        '''
        loss_all = AverageMeter()
        idx = 0
        for data in train_dataloader:
            p = int(data[-1])
            data = data[:-1]
            data = data_extract(data, 5)
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.to(device, dtype=torch.float) for t in data]

            ####################
            # Affine transform
            ####################
            x = data[2]
            y = data[0]
                
            # TODO : Mahesh, check if this is correct or not, just adding another domension should be good ?
            # x = x.unsqueeze(-5)
            # y = y.unsqueeze(-5)
            x_in = torch.cat((x, y), dim=1)
            ct_aff, mat, inv_mats = model(x_in)
            phan = y
            
            # loss = Sim_loss(phan/255, ct_aff/255)
            loss = Sim_loss(phan, ct_aff)
            loss.backward()
            optimizer.step()
            loss_all.update(loss.item(), x.size(0))
        if tb is not None:
            tb.add_scalar("train/sim_loss", loss.item(), epoch)
            
        if save_model and epoch % save_freq==0:
            filename='TransMorph_Affine_e_' + epoch + ".pth.tar"
            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),},
                        'experiments/'+save_dir+filename)
        print('Epoch {}, loss {:.4f}'.format(epoch, loss_all.avg))


    print('=================== Started Testing part =============================')
    # Just run it once, we have saved checkpoint anyway
    for data in test_dataloader:
        data = [t.to(device, dtype=torch.float) for t in data]
        x = data[3]
        y = data[0]
        x = x.unsqueeze(-5)
        y = y.unsqueeze(-5)
        break
    with torch.no_grad():
        model.eval()
        x_in = torch.cat((x, y), dim=1)
        ct_aff, mat, inv_mats = model(x_in)
        phan = y
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(x.cpu().detach().numpy()[0, 0, :, :, 16], cmap='gray')
    plt.title('MR (Moving)')
    plt.subplot(1, 3, 2)
    plt.imshow(y.cpu().detach().numpy()[0, 0, :, :, 16], cmap='gray')
    plt.title('MR (Fixed)')
    plt.subplot(1, 3, 3)
    plt.imshow(ct_aff.cpu().detach().numpy()[0, 0, :, :, 16], cmap='gray')
    plt.title('output')
    print(torch.sum(ct_aff == y))
    print(ct_aff.shape)
    plt.savefig('affine_aligned.png', bbox_inches='tight')