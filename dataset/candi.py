import nibabel
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import logging

def center_crop(x,size):
    ori_size=x.shape
    pad = [int((ori_size[i]-size[i])/2) for i in [0,1,2]]
    y = x[pad[0]:pad[0]+size[0], pad[1]:pad[1]+size[1], pad[2]:pad[2]+size[2]]
    return y

class CANDIDataset(Dataset):
    def __init__(self, mode, datapath, size=[160,160,128]):
        datapath = os.path.expanduser(datapath)
        self.size = size
        self.mode = mode
        self.label = [2,3,4,7,8,10,11,12,13,14,15,16,17,18,24,28,\
            41,42,43,46,47,49,50,51,52,53,54,60]
        self.dice_labels = self.label
        #fix
        for fixpath in os.listdir(f'{datapath}/fix'):
            self.fiximg = self.preprocess_img(f'{datapath}/fix/{fixpath}/anat/NIfTI/anat.nii.gz')
            self.fixseg = self.preprocess_seg(f'{datapath}/fix/{fixpath}/{fixpath}_seg/NIfTI/seg.nii.gz')
        #train/test
        datapath = os.path.join(datapath,mode)
        self.imgpath=[]
        self.segpath=[]
        for subpath in os.listdir(datapath):
            path = os.path.join(datapath, subpath)#subject path
            imgpath = os.path.join(path, 'anat/NIfTI/anat.nii.gz')
            assert os.path.exists(imgpath)
            segpath = os.path.join(path, f'{subpath}_seg/NIfTI/seg.nii.gz')
            assert os.path.exists(segpath)
            self.imgpath.append(imgpath)
            self.segpath.append(segpath)
        
    def __len__(self):
        return len(self.imgpath) 

    def preprocess_img(self, name):
        data = np.array(nibabel.load(name).get_fdata())
        #crop
        data = center_crop(data, self.size)
        #normalize
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        #std_arr = np.sqrt(np.abs(x-mean)/x.size)
        maxp = mean + 6*std
        minp = mean - 6*std
        y = np.clip(data, minp, maxp)
        #import ipdb; ipdb.set_trace()
        #linear transform to [0,1]
        z = (y-y.min())/y.max()
        return z

    def preprocess_seg(self, name):
        data = np.array(nibabel.load(name).get_fdata())
        #crop
        data = center_crop(data, self.size)
        #filter label
        n_class = len(self.label)
        seg = np.zeros_like(data)
        for n,label in enumerate(self.label):
            newlabel = n+1
            seg[data==label]=newlabel
        return seg
            
    def __getitem__(self, idx):
        image = self.preprocess_img(self.imgpath[idx])
        seg = self.preprocess_seg(self.segpath[idx])
        return self.fiximg, self.fixseg, image, seg

def datasplit(rdpath='~/data/CANDI_new', savepth='~/data/CANDI_split', n_fix=1, n_test=20):
    rdpath = os.path.expanduser(rdpath)
    savepth = os.path.expanduser(savepth)
    savefix = os.path.join(savepth,'fix')
    savetr = os.path.join(savepth,'train')
    savetst = os.path.join(savepth,'test')
    sublist = os.listdir(rdpath)
    random.shuffle(sublist)
    for n,sub in enumerate(sublist):
        source = os.path.join(rdpath,sub)
        if n<n_fix:
            target = os.path.join(savefix,sub)
        elif n<n_fix+n_test:
            target = os.path.join(savetst,sub)
        else:
            target = os.path.join(savetr,sub)
        os.system(f'cp -r {source} {target}')


def CANDI_dataloader(args, datapath='~/data/CANDI_split',size=[160,160,128]):
    train_data = CANDIDataset('train', datapath, size=size)
    test_data = CANDIDataset('test',datapath, size=size)
    
    logging.info(f'Train:{len(train_data)}')
    logging.info(f'Test:{len(test_data)}')

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.bsize,
        shuffle=True,
        drop_last= True,
        num_workers=args.num_workers)
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.bsize,
        shuffle=False,
        num_workers=args.num_workers)
    
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    # datapath = '/mnt/data/CANDI_new/SS_084_MR'
    # anat = os.path.join(datapath,'anat/NIfTI/anat.nii.gz')
    # seg = os.path.join(datapath,'SS_084_MR_seg/NIfTI/seg.nii.gz')
    # x=np.array(nibabel.load(anat).get_fdata())
    # y=np.array(nibabel.load(seg).get_fdata())
    # datasplit()
    dataset = CANDIDataset()#checked1mm?
    
    import ipdb; ipdb.set_trace()