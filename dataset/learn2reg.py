import nibabel
import os
import numpy as np
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
import itertools

class Learn2Reg(Dataset):
    def __init__(self, datapath, fconf, size=[192, 160, 192], mod="MR", train=True, test=False):
        self.conf = json.load(open(fconf))
        self.datapath = os.path.expanduser(datapath)
        self.imgpath = []
        self.segpath = []
        self.dice_labels = [0, 1, 2, 3, 4]
        self.size = size
        assert train or test
        assert not (train and test)
        if train:
            subpath_img = "imagesTr"
            subpath_lbl = "labelsTr"
        if test:
            subpath_img = "imagesTs"
            subpath_lbl = "labelsTs"
        if mod == "MR":
            self.filter = "0000"
        else:
            self.filter = "0001"
        for img in os.listdir(os.path.join(self.datapath, subpath_img)):
            if img.split(".")[0].split("_")[-1]==self.filter:
                self.imgpath.append(os.path.join(self.datapath, subpath_img, img))
        
        for lbl in os.listdir(os.path.join(self.datapath, subpath_lbl)):
            if lbl.split(".")[0].split("_")[-1]==self.filter:
                self.segpath.append(os.path.join(self.datapath, subpath_lbl, lbl))
        self.pairs =  list(itertools.product(range(len(self.imgpath)), range(len(self.imgpath))))
    
    def __len__(self):
        return len(self.pairs) 

    def preprocess_img(self, name):
        data = np.array(nibabel.load(name).get_fdata())
        #normalize
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        maxp = mean + 6*std
        minp = mean - 6*std
        y = np.clip(data, minp, maxp)
        z = (y-y.min())
        z = z/z.max()
        return z

    def preprocess_seg(self, name):
        data = np.array(nibabel.load(name).get_fdata())
        return data
            
    def __getitem__(self, idx):
        iidx, fidx =  self.pairs[idx][0], self.pairs[idx][1]
        image = self.preprocess_img(self.imgpath[iidx])
        seg = self.preprocess_seg(self.segpath[iidx])
        fiximg = self.preprocess_img(self.imgpath[fidx])
        fixseg = self.preprocess_seg(self.segpath[fidx])
        seg_fname = "f" + self.imgpath[fidx].split("/")[-1].split(".")[0].split("_")[1]
        seg_fname = seg_fname + "m" + self.imgpath[iidx].split("/")[-1].split(".")[0].split("_")[1]
        return fiximg, fixseg, image, seg, idx, seg_fname

def l2r_dataloader(datapath, size=[192, 160, 192], mod="MR", bsize=1, num_workers=1):
    fconf = datapath + "AbdomenMRCT_dataset.json"
    train_dataset = Learn2Reg(datapath=datapath, fconf=fconf, size=size, mod=mod)
    test_dataset = Learn2Reg(datapath=datapath, fconf=fconf, size=size, mod=mod)
    train_dataloader = DataLoader(train_dataset,
        batch_size=bsize,
        shuffle=True,
        drop_last= True,
        num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset,
        batch_size=bsize,
        shuffle=True,
        drop_last= True,
        num_workers=num_workers)
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    train_dataloader, test_dataloader = l2r_dataloader(datapath="/home/csgrad/mbhosale/Datasets/learn2reg/AbdomenMRCT/")
    for _, samples in enumerate(train_dataloader):
        pass