import torch, sys
from torch.utils.data import Dataset
import dataset.utils.data_utils as data_utils
import numpy as np
from torch.utils.data import DataLoader
import glob
from torchvision import transforms
import dataset.utils.trans as trans


class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path):#, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.dice_labels = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
        # self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = data_utils.pkload(self.atlas_path)
        y, y_seg = data_utils.pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        # Mahesh : Q. Why they have increased the domension here? >> To add Batch size as first dimension and 
        # channel as second dimension, which also is one beacsue the data is gray scale.
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        # x, x_seg = self.transforms([x, x_seg])
        # y, y_seg = self.transforms([y, y_seg])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        # torch.save(x, 'x1.pt')
        # torch.save(y, 'y1.pt')
        return x, x_seg, y, y_seg

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path):#, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        # self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = data_utils.pkload(self.atlas_path)
        y, y_seg = data_utils.pkload(path)
        # Mahesh : Q. Why they have increased the domension here? >> To add Batch size as first dimension and 
        # channel as second dimension, which also is one beacsue the data is gray scale.
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        # x, x_seg = self.transforms([x, x_seg])
        # y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, x_seg, y, y_seg

    def __len__(self):
        return len(self.paths)


def IXI_dataloader(batch_size, num_workers, datapath):
    train_dir = datapath + r'Train/'
    val_dir = datapath + r'Val/'
    atlas_dir = datapath + r'atlas.pkl'
    # train_composed = transforms.Compose([trans.RandomFlip(0),
                                        #  trans.NumpyType((np.float32, np.float32)),
                                        #  ])

    # val_composed = transforms.Compose([trans.Seg_norm(),  # rearrange segmentation label to 1 to 46
                                    #    trans.NumpyType((np.float32, np.int16))])
    train_dir = train_dir + '*.pkl'
    val_dir = val_dir + '*.pkl'
    train_set = IXIBrainDataset(glob.glob(train_dir), atlas_dir)#, transforms=train_composed)
    val_set = IXIBrainInferDataset(glob.glob(val_dir), atlas_dir)#, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader, val_loader
