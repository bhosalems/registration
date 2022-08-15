from logging import root
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import SimpleITK as sitk
import warnings
import random
from pathlib import Path
import math
import numpy as np
import nibabel 
import itertools
import imageio
import multiprocessing

# Chaos is in the DICOM file format but we need to conver it into the nifty file format which we understand
class Chaos_processor(multiprocessing.Process):
    def __init__(self, **kwargs: object) -> None:
        super(Chaos_processor, self).__init__()
        self.fid = kwargs['fid']
        n = kwargs['num_workers']
        n = 20//n
        # temopararily set the 20 becuase there are 20 folders in each CHAOS dataset.
        assert(20%n == 0)
        self.dirs = sorted(os.listdir(kwargs['dicom_path']))[self.fid*n : self.fid*n+n]
        self.dicom_path = kwargs['dicom_path']
        self.modalities = kwargs['modalities']
    
    def run(self):
        self.dicom2nifty(self.modalities)
        self.prepare_seg(self.modalities)

    def dicom2nifty(self, modalities =['T1DUAL', 'T2SPIR']):
        # if os.sep=='\\':
        #     dicom_path = dicom_path.replace('/','\\')
        # elif os.sep == '/':
        #     dicom_path = dicom_path.replace('\\','/')
        # else:
        #     raise ValueError('os.sep must be \\ or /!')
        modalities = modalities
        for dir in self.dirs:
            b_base_record = self.dicom_path + "/" + dir
            if len(modalities):
                for modality in modalities:
                    base_record = b_base_record + "/" +  modality + "/DICOM_anon"
                    if not os.path.exists(base_record):
                        warnings.warn(f"Path {base_record} doesn't exists")
                        continue
                    if modality == "T1DUAL":
                            phases = ["InPhase", "OutPhase"]
                    else:
                        phases = ['']
                    for phase in phases:
                            record = base_record + "/" + phase
                            print("Converting " + record + " ...")
                            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(record)
                            series_reader = sitk.ImageSeriesReader()
                            # Mahesh NOTE: sitk saved the image with the depth as first dimension. But when you read the same
                            # image with nababel with nibabel.load().getfdata()m, it returns the image array with the desired
                            # i.e. with the depth as last dimension.
                            series_reader.SetFileNames(series_file_names)
                            image3D = series_reader.Execute()
                            output_file = "IMG-"+series_file_names[0].split("/")[-1].split("-")[1]
                            output_file = record + "/" + output_file + ".nii.gz"
                            sitk.WriteImage(image3D, output_file)
            else:
                record = b_base_record + "/" +  "DICOM_anon"
                if not os.path.exists(record):
                        warnings.warn(f"Path {record} doesn't exists")
                        continue
                print("Converting " + record + " ...")
                # 1/T1DUAL/DICOM_anon/InPhase"
                series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(record)
                series_reader = sitk.ImageSeriesReader()
                series_reader.SetFileNames(series_file_names)
                image3D = series_reader.Execute()
                output_file = "IMG-"+series_file_names[0].split("/")[-3]
                output_file = record + "/" + output_file + ".nii.gz"
                sitk.WriteImage(image3D, output_file)

    def prepare_seg(self, modalities=['T1DUAL', 'T2SPIR']):
        modalities = modalities
        for dir in self.dirs:
            b_base_record = self.dicom_path + "/" + dir
            if len(modalities):
                for modality in modalities:
                    base_record = b_base_record + "/" +  modality + "/Ground"
                    data_list = []
                    img = None
                    for img in sorted(Path(base_record).rglob("*" + ".png")):
                        data = imageio.imread(img)
                        data = np.transpose(data)
                        data_list.append(data)
                        
                    # Make the depth last dimension
                    data = np.stack(data_list, axis=2)
                    
                    # Mahesh : Q.Determine later if we need mask to make sure everything else other 
                    # than labels in the data is all zeros >> Yes we dont need mask, all are zeros.
                    data = np.where((data>=50) & (data<=70), 1, data) # Liver
                    data = np.where((data>=110) & (data<=135), 2, data) # Right Kidney
                    data = np.where((data>=175) & (data<=200), 3, data) # Left Kidney
                    data = np.where((data>=240) & (data<=255), 4, data) # Spleen
                    
                    # np.save('data.npy', data)
                    
                    # Mahesh : Q. Is this the right way to save the numpy array as the nifty label? >> works now after taking tanspose.
                    seg = nibabel.Nifti1Image(data, affine=np.eye(4))
                    print(img.absolute().as_posix().split("/")[-1].split("-")[1])
                    output_file = "IMG-" + img.absolute().as_posix().split("/")[-1].split("-")[1]
                    output_file = base_record + "/" + output_file + "_seg.nii.gz"
                    nibabel.save(seg, output_file)
            else:
                base_record = b_base_record + "/Ground"
                data_list = []
                img = None
                for img in sorted(Path(base_record).rglob("*" + ".png")):
                    data = imageio.imread(img)
                    # data = np.transpose(data)
                    data_list.append(data)
                    
                # Make the depth last dimension
                data = np.stack(data_list, axis=2)
                
                data = np.where((data>=50) & (data<=70), 1, data) # Liver
                
                # Mahesh : Q. Is this the right way to save the numpy array as the nifty label? >> works now after taking tanspose.
                seg = nibabel.Nifti1Image(data, affine=np.eye(4))
                print(img.absolute().as_posix().split("/")[-3])
                output_file = "IMG-" + img.absolute().as_posix().split("/")[-3]
                output_file = base_record + "/" + output_file + "_seg.nii.gz"
                nibabel.save(seg, output_file)

def datasplit(rdpath='/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR', 
              savepath='/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR', n_fix=1):
    rdpath = os.path.expanduser(rdpath)
    savepath = os.path.expanduser(savepath)
    savefix = os.path.join(savepath,'fix')
    savetr = os.path.join(savepath,'train')
    sublist = os.listdir(rdpath)
    random.shuffle(sublist)
    if not os.path.exists(savefix):
        os.mkdir(savefix)
    if not os.path.exists(savetr):
        os.mkdir(savetr)
    for n,sub in enumerate(sublist):
        source = os.path.join(rdpath, sub)
        if n<n_fix:
            target = os.path.join(savefix, sub)
        else:
            target = os.path.join(savetr, sub)
        os.system(f'cp -r {source} {target}')
        

# TODO: Look how to retireve labels from the segmentaiton png image we got.
class ChaosDataset(Dataset):
    def __init__(self, datapath, size, modality, phase, pad, ext=".nii.gz", label_ext=".png", calc_dice=True, num_samples=0) -> None:
        datapath = os.path.expanduser(datapath)
        self.size = size
        self.mode = modality
        self.ext = ext
        self.label_ext = label_ext
        self.pad = pad
        self.dice_labels = [0, 1, 2, 3, 4]
        self.downsample_rate = 16
        # Mahesh : Q. Wasnt able to modify the the funcs list (for storeing the affines of images just for reconstructing images for verification) 
        # outside init, e.g. in __get_item__(), Why? It was empty.
        # self.funcs = []

        #train/test
        self.imgpath = []
        self.segpath = []
        for subpath in os.listdir(datapath):
            path = os.path.join(datapath, subpath)
            path = os.path.join(path, f"{modality}/DICOM_anon/{phase}")
            assert os.path.exists(path)
            self.imgpath.append(path)
            if calc_dice:
                path = datapath + f'/{subpath}/{modality}/Ground'
                assert os.path.exists(path)
                self.segpath.append(path)
        assert(len(self.imgpath)!=0)
        n = len(self.imgpath)
        pairs = itertools.product(range(n),range(n))
        self.pairs = list(pairs)
        random.shuffle(self.pairs)
        if num_samples != 0:
            self.num_samples = num_samples
        else:
            self.num_samples = len(self.pairs)
        
        
    def zero_pad(self, data, value=0):
        orig_size = data.shape
        c_dim = orig_size[-1]
        pad_sz = abs(c_dim - (math.ceil(c_dim/self.downsample_rate)*self.downsample_rate))
        data = torch.nn.functional.pad(data, (math.floor(pad_sz/2), math.ceil(pad_sz/2)), value=value)
        assert(data.shape[-1]%self.downsample_rate==0)
        return data
    
    def preprocess_img(self, data_path, pad, pad_sz):
        """Preprocess the nii.gz images, we need to pad to the given size when required

        Args: 
            data_path (string): directory containing the path of nii.gz files to be preprocessed
            pad (boolean): if pad r not
            pad_sz (iterable) : desired size after padding
        """
        # Since, Path().rglob returns the generator and converting the generator to the list is not stable, just iterating anyway,
        # it just returns the signle image.
        data = None
        for img in Path(data_path).rglob("*" + self.ext):
            data = nibabel.load(img)
            data = np.array(data.get_fdata())
            if pad:
                pad_w = -data.shape[0] + pad_sz[0]
                pad_h = -data.shape[1] + pad_sz[1]
                pad_d = -data.shape[2] + pad_sz[2]
                assert((pad_w >= 0 and pad_h >= 0 and pad_d >=0))
                
                # Mahesh : Q. Why was it called fixed_nopad before? Am I missing mosmething, it appears it should be same for both the 
                # fixed and the moving image.
                nopad = np.ones(data.shape)
                data = np.pad(data, ((math.floor(pad_w/2), math.ceil(pad_w/2)), (math.floor(pad_h/2), math.ceil(pad_h/2)), 
                              (math.floor(pad_d/2), math.ceil(pad_d/2))), 'constant')
                nopad = np.pad(nopad, ((math.floor(pad_w/2), math.ceil(pad_w/2)), (math.floor(pad_h/2), math.ceil(pad_h/2)), 
                              (math.floor(pad_d/2), math.ceil(pad_d/2))), 'constant')
        
       
        # Mahesh : Q. Should we normalize the images ? What other processing is needed here?
        #normalize
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        maxp = mean + 6*std
        minp = mean - 6*std
        y = np.clip(data, minp, maxp)
        z = (y-y.min())/y.max()
        data = z
        data = np.ascontiguousarray(data)
        data = torch.from_numpy(data)
        nopad = np.ascontiguousarray(nopad)
        nopad = torch.from_numpy(nopad)
        if data.shape[-1]%self.downsample_rate != 0:
            data = self.zero_pad(data)
            nopad = self.zero_pad(nopad)
            assert(data.shape == nopad.shape)
            # seg = self.zero_pad(seg)
            # Mahesh : Q. Does this really update the self.size?
            self.size = data.shape
        return data, nopad
    
    def __getitem__(self, index):
        i = self.pairs[index]
        movingimg, moving_nopad = self.preprocess_img(self.imgpath[i[0]], pad=self.pad, pad_sz=self.size)
        fixedimg, fixed_nopad = self.preprocess_img(self.imgpath[i[1]], pad=self.pad, pad_sz=self.size)
        assert(movingimg.shape==fixedimg.shape)
        if len(self.segpath)!=0:
            moving_seg = self.preprocess_seg(self.segpath[i[0]], pad=self.pad, pad_sz=self.size)
            fixed_seg = self.preprocess_seg(self.segpath[i[1]], pad=self.pad, pad_sz=self.size)
            assert(fixed_seg.shape==moving_seg.shape)
            return fixedimg, fixed_seg, fixed_nopad, movingimg, moving_seg, index 
        return fixedimg, fixed_nopad, movingimg, index
    
    def __len__(self):
        return self.num_samples
    
    def preprocess_seg(self, data_path, pad, pad_sz):
        """Preprocess the .png segment labels, we need to pad to the given size when required

        Args:
            data_path (string): directory containing the path of nii.gz files to be preprocessed
            pad (boolean): if pad r not
            pad_sz (iterable) : desired size after padding
        """
        for img in Path(data_path).rglob("*" + self.ext):
            data = nibabel.load(img)
            data = np.array(data.get_fdata())       
            if pad:
                pad_w = -data.shape[0] + pad_sz[0]
                pad_h = -data.shape[1] + pad_sz[1]
                pad_d = -data.shape[2] + pad_sz[2]
                assert(pad_w >= 0 and pad_h >= 0 and pad_d >=0)
                data = np.pad(data, ((math.floor(pad_w/2), math.ceil(pad_w/2)), 
                                     (math.floor(pad_h/2), math.ceil(pad_h/2)), 
                                (math.floor(pad_d/2), math.ceil(pad_d/2))), 'constant')
        data = torch.from_numpy(data)
        return data

def Chaos_dataloader(root_path, bsize, tr_path, tst_path, tr_modality, tr_phase, tst_modality, tst_phase, 
                     size=[400, 400, 50], data_split=False, n_fix=1, num_workers = 1, augment=True, pad=True, 
                     tr_num_samples=None, tst_num_samples=10):
    if(data_split):
        train_rootpath = root_path + 'CHAOS_Train_Sets/Train_Sets/MR'
        validation_rootpath = root_path + 'CHAOS_Test_Sets/Test_Sets/MR'
        train_savepath = train_rootpath
        validation_savepath = validation_rootpath
        datasplit(train_rootpath, train_savepath, n_fix) 
        datasplit(validation_rootpath, validation_savepath, n_fix) 

    train_dataset = ChaosDataset(datapath=tr_path, size=size, pad=pad, modality=tr_modality, phase=tr_phase, calc_dice=True, num_samples=tr_num_samples)
    # TODO : Mahesh : Currently due to rules of the competition they did not release the segmentation data of the
    # /data_local/mbhosale/CHAOS/CHAOS_Test_Sets/Test_Sets/MR/, But we could still calculate the similarity loss anyway, for 
    # calculation of the dice score we will need the Ground truth, therefore I will randomly divided some data from Train_Sets
    # as test, later once we are satified with the performance on the similarity loss. >> No we have to use different
    # different modalities here, so we dont have to divide the train data, we could use T2modality train data.
    test_dataset = ChaosDataset(datapath=tst_path, size=size, pad=pad, modality=tst_modality, phase=tst_phase, calc_dice=True, num_samples=tst_num_samples)
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
    data_path = r"/data_local/mbhosale/CHAOS/"
    # dicom2nifty(r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR")
    # dicom2nifty(r"/data_local/mbhosale/CHAOS/CHAOS_Test_Sets/Test_Sets/MR")
    # dicom2nifty(r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/CT", modalities=[])
    # dicom2nifty(r"/data_local/mbhosale/CHAOS/CHAOS_Test_Sets/Test_Sets/CT", modalities=[])
    
    # prepare_seg(r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR")
    # prepare_seg(r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/CT", modalities=[])
    # NOTE There's no ground truth in the test images.
    
    c1 = Chaos_processor(dicom_path=r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR", modalities=['T1DUAL', 'T2SPIR'], 
                        num_workers=5, fid=0)
    c2 = Chaos_processor(dicom_path=r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR", modalities=['T1DUAL', 'T2SPIR'], 
                        num_workers=5, fid=1)
    c3 = Chaos_processor(dicom_path=r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR", modalities=['T1DUAL', 'T2SPIR'], 
                        num_workers=5, fid=2)
    c4 = Chaos_processor(dicom_path=r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR", modalities=['T1DUAL', 'T2SPIR'], 
                        num_workers=5, fid=3)
    c5 = Chaos_processor(dicom_path=r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR", modalities=['T1DUAL', 'T2SPIR'], 
                        num_workers=5, fid=4)
    c1.start()
    c2.start()
    c3.start()
    c4.start()
    c5.start()
    c1.join()
    c2.join()
    c3.join()
    c4.join()
    c5.join()
    
    c1 = Chaos_processor(dicom_path=r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/CT", modalities=[], 
                        num_workers=5, fid=0)
    c2 = Chaos_processor(dicom_path=r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/CT", modalities=[], 
                        num_workers=5, fid=1)
    c3 = Chaos_processor(dicom_path=r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/CT", modalities=[], 
                        num_workers=5, fid=2)
    c4 = Chaos_processor(dicom_path=r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/CT", modalities=[], 
                        num_workers=5, fid=3)
    c5 = Chaos_processor(dicom_path=r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/CT", modalities=[], 
                        num_workers=5, fid=4)
    c1.start()
    c2.start()
    c3.start()
    c4.start()
    c5.start()
    c1.join()
    c2.join()
    c3.join()
    c4.join()
    c5.join()
    
    # We won't be splitting data permamenently we rather choose the pairs of fixed and moving images, similar to msd dataloader.
    # datasplit(rdpath='/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR', 
            #   savepath='/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR')
    # datasplit(rdpath='/data_local/mbhosale/CHAOS/CHAOS_Test_Sets/Test_Sets/MR', 
            #   savepath='/data_local/mbhosale/CHAOS/CHAOS_Test_Sets/Test_Sets/MR')
    
    # We will pad the images to max H,W,D i.e. 400x400x50
    # train_dataloader, test_dataloader = Chaos_dataloader(root_path=data_path,  tr_path=r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR/", 
    #                                                      tst_path=r"/data_local/mbhosale/CHAOS/CHAOS_Train_Sets/Train_Sets/MR/", 
    #                                                      bsize=1, tr_modality='T1DUAL', tr_phase='InPhase', tst_modality='T1DUAL', 
    #                                                      tst_phase='OutPhase', size=[400, 400, 50], data_split=False, n_fix=1)
    # for i, samples in enumerate(train_dataloader):
    #     print(samples[0].shape)
    #     # ni_img = nibabel.Nifti1Image(train_dataloader.dataset.fiximg, train_dataloader.dataset.func)
    #     # nibabel.save(ni_img, 'output.nii.gz')
    
    # for _, samples in enumerate(test_dataloader):
    #     print(samples[0].shape)
