import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import nibabel
import torch

def center_crop(x,size):
    ori_size=x.shape
    pad = [int((ori_size[i]-size[i])/2) for i in [0,1,2]]
    y = x[pad[0]:pad[0]+size[0], pad[1]:pad[1]+size[1], pad[2]:pad[2]+size[2]]
    return y

class BraTSDataset(Dataset):
    def __init__(self, data_path, mod, seg='seg', size=[240, 240, 155]):
        """ Dataset for https://www.med.upenn.edu/sbia/brats2018/data.html 

        Args:
            data_path (str): Root path of the dataset
            mod (str): Modality of the data
            size (list): image size
        """

        self.size = size
        self.datapath = os.path.expanduser(data_path)
        self.mod = mod
        self.seg = seg
        self.labels = [0, 1, 2, 4]
        self.dice_labels = self.labels

        # fix
        for fixpath in os.listdir(f'{self.datapath}/fix'):
            for f in os.listdir(f'{self.datapath}/fix/{fixpath}'):
                if self.mod in f:
                    self.fiximg = self.preprocess_img(f'{self.datapath}/fix/{fixpath}/{f}')
                    self.fiximg = self.fiximg[None, ...]
                    self.fiximg = np.ascontiguousarray(self.fiximg)
                    self.fiximg= torch.from_numpy(self.fiximg)
                if self.seg in f:
                    self.fixseg = self.preprocess_seg(f'{self.datapath}/fix/{fixpath}/{f}')
                    # self.fixseg = self.fixseg[None, ...]
                    self.fixseg = np.ascontiguousarray(self.fixseg)
                    self.fixseg = torch.from_numpy(self.fixseg)
        
        # Train and Test Data
        self.imgpath = []
        self.segpath = []
        for subpath in os.listdir(f'{self.datapath}/data'):
            path = os.path.join(f'{self.datapath}/data', subpath)#subject path
            for f in os.listdir(path):
                if self.mod in f:
                    imgpath = os.path.join(path, f)
                    assert os.path.exists(imgpath)
                    self.imgpath.append(imgpath)
                elif self.seg in f:
                    segpath = os.path.join(path, f)
                    assert os.path.exists(segpath)
                    self.segpath.append(segpath)
    
    def __getitem__(self, idx):
        image = self.preprocess_img(self.imgpath[idx])
        seg = self.preprocess_seg(self.segpath[idx])
        image = image[None, ...]
        image = np.ascontiguousarray(image)
        seg = np.ascontiguousarray(seg)
        image, seg = torch.from_numpy(image), torch.from_numpy(seg)
        return self.fiximg, self.fixseg, image, seg 
    
    def __len__(self):
        return len(self.imgpath) 
        
    def preprocess_img(self, name):
        """Reads and preprocesses the image. First crop at centre, then clip the image in range 
        [mean + 6*std, mean - 6*std]. At lasr, normalize the image.

        Args:
            name (str): filepath

        Returns:
            array: Preprocesses image array
        """
        data = np.array(nibabel.load(name).get_fdata())
        # crop
        # TODO what should be the size of the image when cropping.
        data = center_crop(data, [240, 240, 144])
        self.size = [240, 240, 144]
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
        # crop
        # TODO what should be the size of the segmentation when cropping.
        data = center_crop(data, [240, 240, 144])
        n_class = len(self.labels)
        seg = np.zeros_like(data)
        mapping = {0:1, 1:2, 2:3, 4:4}
        for _,label in enumerate(self.labels):
            newlabel = mapping[label]
            seg[data==label] = newlabel
        return seg


def datasplit(rdpath, savepth='/data_local/xuangong/data/BraTS/BraTS2018/new', n_fix=1):
    """Splits the braTS dataset into fix and data part. Fix contains the volumes used as fixed image
    and data contains all the moving volumes. NOTE This method should be called only once at the start.

    Args:
        rdpath (str): path of the root where data is stored
        savepth (str, optional): Where to save the data after splitting in fix and data. New folders 
        fix and data will becreated inside this folder.. Defaults to '/data_local/xuangong/data/BraTS/BraTS2018/new'.
        n_fix (int, optional): How many image volumes from the dataset to be considered as fixed. Defaults to 1.
    """
    rdpath = os.path.expanduser(rdpath)
    savepth = os.path.expanduser(savepth)
    savefix = os.path.join(savepth,'fix')
    savetr = os.path.join(savepth, 'data')
    sublist = os.listdir(rdpath)
    random.shuffle(sublist)
    for n, sub in enumerate(sublist):
        source = os.path.join(rdpath, sub)
        if n < n_fix:
            target = os.path.join(savefix, sub)
        else:
            target = os.path.join(savetr,sub)
        os.system(f'cp -r {source} {target}')


def braTS_dataloader(root_path, save_path, bsize, mod, seg = "seg", size=[240, 240, 155], data_split=False, n_fix=1, num_workers = 4):
    if(data_split):
        train_rootpath = root_path + '_Train'
        validation_rootpath = root_path + '_Validation'
        train_savepath = os.path.join(save_path, "Train")
        validation_savepath = os.path.join(save_path, "Validation")
        datasplit(train_rootpath, train_savepath, n_fix) 
        datasplit(validation_rootpath, validation_savepath, n_fix) 

    tr_path = os.path.join(save_path, "Train")
    ts_path = os.path.join(save_path, "Validation")

    train_dataset = BraTSDataset(tr_path, mod)
    test_dataset =  BraTSDataset(ts_path, mod)

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
    root_path = "/data_local/xuangong/data/BraTS/BraTS2018"
    save_path = "/home/csgrad/mbhosale/Image_registration/datasets/BraTS2018/"

    # Todo : Only dummy size, need to define this later after checking
    pad_size = [240, 240, 155]
    mod = "flair"
    bsize = 1

    # Need to call data_split only once for train as well as validation at first.
    train_dataloader, test_dataloader = braTS_dataloader(root_path, save_path, bsize, mod)

    for _, samples in enumerate(train_dataloader):
        print(samples.shape)
    
    for _, samples in enumerate(test_dataloader):
        print(samples.shape)