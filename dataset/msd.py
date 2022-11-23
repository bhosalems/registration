import nibabel
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import itertools
import math
import random
import SimpleITK as sitk
import cv2

import logging
# setting path
import sys
sys.path.append('../')
import losses
from dataset.transform import random_transform, random_transform_elastic

def NIISplit(img_path, label_path, pad_size, dataset = 'hippocampus', mode = 'vm', train = 0, trainseg=0, valseg = 0, valreg = 0, 
        regmode = 'sample', tr_percent=1, bootstrap_prop=1, divide =None):
    img_path = os.path.expanduser(img_path)
    label_path = os.path.expanduser(label_path)

    if dataset=='hippocampus':
        test_index = [349, 251, 50, 363, 197, 96, 334, 345, 355, 298, 232, 49, 205, \
            338, 101, 38, 311, 223, 390, 204, 221, 350, 180, 276, 177, 124]
    elif dataset == 'prostate':
        test_index = [16, 4, 32, 20, 43, 18, 6, 1]
    elif dataset == "liver":
        test_index = [10, 33, 41, 67, 98, 123, 114, 79, 82, 55]
    # 
    test_filenames = []
    train_filenames = []
    for file in os.listdir(img_path):
        if not file.endswith("nii.gz"):
            continue
        #
        idx = int(file.split('_')[1].split('.')[0])
        isTest = idx in test_index
        if isTest:
            test_filenames.append(file)
        else:
            train_filenames.append(file)
    # import ipdb; ipdb.set_trace()
    if trainseg:
        train_seg_data = NIIDatasetTestSeg(imgpath=img_path, labelpath =label_path, 
            test_names=train_filenames, train_names=train_filenames, padsize = pad_size)
        return train_seg_data

    train_data, train_seg_data, val_seg_data, val_reg_data = None, None, None, None
    if valseg:
        val_seg_data = NIIDatasetTestSeg(imgpath=img_path, labelpath =label_path, divide = divide,
            test_names=test_filenames, train_names=train_filenames, padsize = pad_size)
    if valreg:
        val_reg_data = NIIDatasetTestReg(imgpath=img_path, labelpath =label_path, divide = divide,
            test_names=test_filenames, mode = regmode, padsize = pad_size)
        print('ValReg',len(val_reg_data))

    if train:
        if bootstrap_prop!=1:
            N_ = int(bootstrap_prop * len(train_filenames))
            train_filenames_subset = np.random.choice(train_filenames, size =  N_, replace = True)
            train_filenames = list(train_filenames_subset)
        # import ipdb; ipdb.set_trace()
        
        train_data = NIIDatasetPaired(imgpath=img_path, labelpath =label_path, divide = divide,
            filenames = train_filenames, mode = mode, tr_percent=tr_percent, padsize = pad_size)
    
    
    return train_data, val_seg_data, val_reg_data
  

def MSD_dataloader(dataset, bsize, num_workers, pad_size, datapath='~/data/MSD', tr_percent=1, testseg=1, testreg=0):
    imgpath = f'{datapath}/{dataset}/all/data'
    labelpath =  f'{datapath}/{dataset}/all/labels'
    
    train_data, val_seg_data, val_reg_data = NIISplit(
        imgpath, labelpath, pad_size, mode = 'vm', train=1, valseg=testseg, valreg=testreg, divide = None,
        tr_percent = tr_percent, dataset = dataset, bootstrap_prop=1)
    
    # import ipdb; ipdb.set_trace()
    logging.info(f'Train pairs:{len(train_data)}')
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=bsize,
        shuffle=True,
        drop_last= True,
        num_workers=num_workers)
    valseg_dataloader, valreg_dataloader = None, None
    if testseg:
        valseg_dataloader = torch.utils.data.DataLoader(
            val_seg_data,
            batch_size=bsize,
            shuffle=True,
            num_workers=num_workers)
    if testreg:
        valreg_dataloader = torch.utils.data.DataLoader(
            val_reg_data,
            batch_size=bsize,
            shuffle=True,
            num_workers=num_workers) 
    return train_dataloader, valseg_dataloader, valreg_dataloader

def MSD_test_dataloader(dataset, bsize, pad_size, datapath='~/data/MSD', testseg=1, testreg=0):
    imgpath = f'{datapath}/{dataset}/all/data'
    labelpath =  f'{datapath}/{dataset}/all/labels'
    
    train_data, val_seg_data, val_reg_data = NIISplit(
        imgpath, labelpath, pad_size, mode = 'vm', train=0, valseg=testseg, valreg=testreg, divide = None,
        tr_percent = 0, dataset = dataset, bootstrap_prop=1)
    
    # import ipdb; ipdb.set_trace()
    valseg_dataloader, valreg_dataloader = None, None
    if testseg:
        valseg_dataloader = torch.utils.data.DataLoader(
            val_seg_data,
            batch_size=bsize,
            shuffle=True,
            num_workers=1)
    if testreg:
        valreg_dataloader = torch.utils.data.DataLoader(
            val_reg_data,
            batch_size=bsize,
            shuffle=True,
            num_workers=1) 
    return valseg_dataloader, valreg_dataloader

class NIIDatasetPaired(Dataset):
    def __init__(self, imgpath=None, labelpath=None, filenames = None, mode ='vm', initdata = False, tr_percent=1, padsize=(48,64,48), divide=None):
        self.imgpath = imgpath
        self.labelpath = labelpath
        self.filenames = filenames
        n = len(self.filenames)
        path_names = self.imgpath.split("/")
        if "liver" in path_names:
            self.dice_labels = [0, 1, 2]
        else:
            self.dice_labels = [0, 1, 2, 3]
        if tr_percent<1:
            # n = int(n*tr_percent)
            n=4
        pairs = itertools.product(range(n),range(n))
        self.pairs = list(pairs)
        random.shuffle(self.pairs)
        # import ipdb; ipdb.set_trace()
        # if (divide is not None) and (self.pairs is not None):
        if (divide is not None):
            max_below = divide*(len(self.pairs)//divide)
            logging.info(f'{len(self.pairs)}divide by{divide} to get {max_below}')
            self.pairs = self.pairs[:max_below]
        
        # import ipdb; ipdb.set_trace()
        self.mode = mode
        self.initdata = initdata
        self.datadict=[]
        self.pad_size = padsize
        if self.initdata:
            for idx in range(n):
                data_path = os.path.join(self.imgpath, self.filenames[idx])
                label_path = os.path.join(self.labelpath, self.filenames[idx])
                data, nopad = preprocess(data_path, isimg = True, padsize=self.pad_size)
                label, _ = preprocess(label_path, isimg = False, padsize=self.pad_size)
                savedic = {}
                savedic['data'] = data
                savedic['nopad'] = nopad
                savedic['label'] = label
                self.datadict.append(savedic)

    def __len__(self):
        return len(self.pairs) 

    def __getitem__(self, idx):
        i = self.pairs[idx]
        if self.initdata:
            fix  = self.datadict[i[0]]
            moving = self.datadict[i[1]]
            fixed_data = fix['data']
            fix_nopad = fix['nopad']
            fixed_label = fix['label']
            moving_data = moving['data']
            moving_label = moving['label']
        else:
            f1 = self.filenames[i[0]]
            seg_fname = "f"+ self.filenames[i[0]].split('.')[-3].split("_")[-1]
            # TODO Mahesh : Add some good heuristic or other logic to choose the fixed image, the MSD liuver dataset is huge
            # with arround 121 total volumes and therefore 121* 121 ~ 14000 samples ~ 10 days of mere training. 
            f2 = self.filenames[i[1]]
            seg_fname = seg_fname + "m"+ self.filenames[i[0]].split('.')[-3].split("_")[-1]
            
            data_path = os.path.join(self.imgpath,f1)
            label_path = os.path.join(self.labelpath,f1)
            fixed_data, fix_nopad = preprocess(data_path, isimg = True, padsize=self.pad_size)
            fixed_label, _ = preprocess(label_path, isimg = False, padsize=self.pad_size)

            data_path = os.path.join(self.imgpath,f2)
            label_path = os.path.join(self.labelpath,f2)
            moving_data, moving_nopad = preprocess(data_path, isimg = True, padsize=self.pad_size)
            moving_label, _ = preprocess(label_path, isimg = False, padsize=self.pad_size)
        if self.mode == 'vm':
            return fixed_data, fixed_label, fix_nopad, moving_data, moving_label, idx, seg_fname
        else:
            fix2, fix2_label, fix2_nopad, displacement = random_transform(
                moving_data, moving_label, moving_nopad, rot=1, scale=1, translate=1)
            return fixed_data, fixed_label, fix_nopad, moving_data, moving_label, fix2, fix2_label, fix2_nopad, displacement

        
        # fixed_data2, fixed_label2, displacement = random_transform(self.moving_data, self.moving_label)
        # return fixed_data1, fixed_label1, fixed_data2, fixed_label2, self.moving_data, self.moving_label, displacement

class NIIDatasetTestReg(Dataset):
    def __init__(self, imgpath=None, labelpath=None, test_names=None, mode='trans', padsize=(48,64,48), divide=None):
        self.imgpath = imgpath
        self.labelpath = labelpath
        self.filenames = test_names
        self.mode = mode
        n = len(self.filenames)
        pairs = itertools.product(range(n),range(n))
        self.pairs = list(pairs)
        random.shuffle(self.pairs)
        
        path_names = self.imgpath.split("/")
        if "liver" in path_names:
            self.dice_labels = [0, 1, 2]
        else:
            self.dice_labels = [0, 1, 2, 3]
            
        # import ipdb; ipdb.set_trace()
        # if (divide is not None) and (self.pairs is not None):
        if (divide is not None):
            max_below = divide*(len(self.pairs)//divide)
            logging.info(f'{len(self.pairs)}divide by{divide} to get {max_below}')
            self.pairs = self.pairs[:max_below]
        self.pad_size = padsize

    def __len__(self):
        return len(self.pairs) 

    def __getitem__(self,idx):
        # NOTE with this option, we cant trust on the segment image file names, as fixed image is just the transform of the moving image.
        if self.mode == 'trans':
            f1 = self.filenames[idx]
            # fix, fix_idx= preprocess(os.path.join(self.imgpath, f1), isimg=True)
            # fixlabel, _ = preprocess(os.path.join(self.labelpath, f1), isimg=False)
            # moving_data, moving_label, displacement = random_transform(fix, fixlabel, rot=1, scale=1, translate=1)#dis: fix->moving

            moving_data, moving_nopad= preprocess(os.path.join(self.imgpath, f1), isimg=True, padsize=self.pad_size)
            moving_label, _ = preprocess(os.path.join(self.labelpath, f1), isimg=False, padsize=self.pad_size)
            fix, fixlabel, fix_nopad, displacement = random_transform(moving_data, moving_label, moving_nopad, rot=1, scale=1, translate=1)

            return fix, fixlabel, fix_nopad, moving_data, moving_label, displacement, idx
        else:
            i = self.pairs[idx]
            f1 = self.filenames[i[0]]
            f2 = self.filenames[i[1]]
            seg_fname = "f"+ f1.split('.')[-3].split("_")[-1]
            seg_fname = seg_fname + "m"+ f2.split('.')[-3].split("_")[-1]
            fix_data, fix_nopad= preprocess(os.path.join(self.imgpath, f1), isimg=True, padsize=self.pad_size)
            fix_label, _ = preprocess(os.path.join(self.labelpath, f1), isimg=False, padsize=self.pad_size)

            moving_data, _= preprocess(os.path.join(self.imgpath, f2), isimg=True, padsize=self.pad_size)
            moving_label, _ = preprocess(os.path.join(self.labelpath, f2), isimg=False, padsize=self.pad_size)
            return fix_data, fix_label, fix_nopad, moving_data, moving_label, idx, seg_fname

  
class NIIDatasetTestSeg(Dataset):
    def __init__(self, imgpath=None, labelpath=None, test_names=None, train_names=None, padsize=(48,64,48), divide=None):
        self.imgpath = imgpath
        self.labelpath = labelpath
        self.filenames = test_names
        self.corres_names = []
        # self.train_names = train_names
        self.pad_size = padsize
        
        path_names = self.imgpath.split("/")
        if "liver" in path_names:
            self.dice_labels = [0, 1, 2]
        else:
            self.dice_labels = [0, 1, 2, 3]
            
        # TODO Mahesh: Commenting it for some time becuase we do not want to choose the fixed image 
        # from the train data. >> It is possible to use such setting for the testing.
        for filename in test_names:
            data, _ = preprocess(os.path.join(self.imgpath, filename), isimg=True, padsize=self.pad_size)
            # print(np.max(data))
            # print(np.min(data))
            ts_1 = torch.unsqueeze(torch.unsqueeze( torch.from_numpy(data), 0), 0)
            ncc_best = 0
            for tr_name in train_names:
                tr_data, _ = preprocess(os.path.join(self.imgpath, tr_name), isimg=True, padsize=self.pad_size)
                ts_2 = torch.unsqueeze(torch.unsqueeze( torch.from_numpy(tr_data), 0), 0)
                assert ts_1.max()<=1
                assert ts_2.max()<=1
                assert ts_1.min()>=0
                assert ts_2.min()>=0

                ncc = losses.ncc_loss(ts_1.float().cuda(), ts_2.float().cuda())#[0,1]
                if ncc<ncc_best:
                    ncc_best = ncc
                    corr_name = tr_name
            self.corres_names.append(corr_name)

    def __len__(self):
        return len(self.filenames) 

    def __getitem__(self,idx):
        f1 = self.filenames[idx]
        f2 = self.corres_names[idx]
        seg_fname = "f"+ f2.split('.')[-3].split("_")[-1]
        seg_fname = seg_fname + "m"+ f1.split('.')[-3].split("_")[-1]
        moving_data, _ = preprocess(os.path.join(self.imgpath, f1), isimg=True, padsize=self.pad_size)
        moving_label, _ = preprocess(os.path.join(self.labelpath, f1), isimg=False, padsize=self.pad_size) 

        fix_data, fix_nopad= preprocess(os.path.join(self.imgpath, f2), isimg=True, padsize=self.pad_size)
        fix_label, _ = preprocess(os.path.join(self.labelpath, f2), isimg=False, padsize=self.pad_size) 

        return fix_data,  fix_label, fix_nopad, moving_data, moving_label, idx, seg_fname


class NIIDatasetAllTestSeg(Dataset):
    def __init__(self, imgpath=None, labelpath=None, test_names=None, train_names=None, padsize=(48,64,48), divide=None):
        self.imgpath = imgpath
        self.labelpath = labelpath
        self.filenames = test_names
        self.corres_names = []
        #self.train_names = train_names
        self.pad_size = padsize
        path_names = self.imgpath.split("/")
        if "liver" in path_names:
            self.dice_labels = [0, 1, 2]
        else:
            self.dice_labels = [0, 1, 2, 3]
            
        # import ipdb; ipdb.set_trace()
        for filename in test_names:
            data, _ = preprocess(os.path.join(self.imgpath, filename), isimg=True, padsize=self.pad_size)
            ts_1 = torch.unsqueeze(torch.unsqueeze( torch.from_numpy(data), 0), 0)
            ncc_best = 0
            for tr_name in train_names:
                tr_data, _ = preprocess(os.path.join(self.imgpath, tr_name), isimg=True, padsize=self.pad_size)
                ts_2 = torch.unsqueeze(torch.unsqueeze( torch.from_numpy(tr_data), 0), 0)
                assert ts_1.max()<=1
                assert ts_2.max()<=1
                assert ts_1.min()>=0
                assert ts_2.min()>=0

                ncc = losses.ncc_loss(ts_1.float().cuda(), ts_2.float().cuda())#[0,1]
                if ncc<ncc_best:
                    ncc_best = ncc
                    corr_name = tr_name
            self.corres_names.append(corr_name)

    def __len__(self):
        return len(self.filenames) 

    def __getitem__(self,idx):
        f1 = self.filenames[idx]
        f2 = self.corres_names[idx]
        seg_fname = "f"+ f2.split('.')[-3].split("_")[-1]
        seg_fname = seg_fname + "m"+ f1.split('.')[-3].split("_")[-1]

        moving_data, _ = preprocess(os.path.join(self.imgpath, f1), isimg=True, padsize=self.pad_size)
        moving_label, _ = preprocess(os.path.join(self.labelpath, f1), isimg=False, padsize=self.pad_size) 

        fix_data, fix_nopad= preprocess(os.path.join(self.imgpath, f2), isimg=True, padsize=self.pad_size)
        fix_label, _ = preprocess(os.path.join(self.labelpath, f2), isimg=False, padsize=self.pad_size) 

        return fix_data,  fix_label, fix_nopad, moving_data, moving_label, idx, seg_fname


def pad(x, shape):
    # import ipdb; ipdb.set_trace()

    s_x = math.floor((shape[0] - x.shape[0])/2)
    s_y = math.floor((shape[1] - x.shape[1])/2)
    s_z = math.floor((shape[2] - x.shape[2])/2)
    new_x = np.zeros(shape)
    new_x[s_x:s_x+x.shape[0],s_y: s_y + x.shape[1], s_z:s_z + x.shape[2]] = x
    #save_index = [s_x, x.shape[0], s_y, x.shape[1], s_z, x.shape[2]]
    nopad = np.zeros_like(new_x)
    nopad[s_x:s_x+x.shape[0],s_y: s_y + x.shape[1], s_z:s_z + x.shape[2]] = 1
    return new_x, nopad


def normalize(x):
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    maxp = mean + 6*std
    minp = mean - 6*std
    y = np.clip(x, minp, maxp)
    z = (y-y.min())
    z = z / z.max()
    return z


# # python function replica of matlab's mat2gray
def matlab_mat2grey(A = False):
    alpha = min(A.flatten())
    beta = max(A.flatten())
    I = A
    cv2.normalize(A, I, alpha , beta ,cv2.NORM_MINMAX)
    I = np.uint8(I)
    return I


def preprocess(name, padsize = (48,64,48), isimg=False): #resample=[1,1,1], 
    image = np.array(nibabel.load(name).get_fdata())
    # image = image - image.min()
    # A = np.double(image)
    # out = np.zeros(A.shape, np.double)
    # normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)
    # normalized = matlab_mat2grey(image)
    # normalized = sitk.GetImageFromArray(normalized)
    # sitk.WriteImage(normalized, 'test_image.nii.gz')
    if isimg:
        image = normalize(image)
        # normalized = sitk.GetImageFromArray(image)
        # sitk.WriteImage(normalized, 'test_image.nii.gz')
    image, nopad = pad(image, padsize)
    return image, nopad


if __name__ == "__main__":
    datapath = '~/data/hippocampus/all'
    HippoSplit(datapath)