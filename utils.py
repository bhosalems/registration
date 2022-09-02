import torch
import logging
import nibabel as nib
import numpy as np
import os
from PIL import Image
from pathlib import Path
import imageio
import SimpleITK as sitk
import nibabel as nib

def tensor2nii(pred, true, fnames, mode, one_hot=True, flow=None):
        assert(len(fnames)>=2)        
        assert(len(pred.shape)==5)
        assert(len(true.shape)==5)

        if mode == 'seg':
            if one_hot:
                true = torch.max(true.detach().cpu(), dim=1)[1]
                pred = torch.max(pred.detach().cpu(), dim=1)[1]
            true = true[0, ...].numpy().astype(np.uint8)
            pred = pred[0, ...].numpy().astype(np.uint8)
            
            true = np.where((true>=50) & (true<=70), 1, true) # Liver
            true = np.where((true>=110) & (true<=135), 2, true) # Right Kidney
            true = np.where((true>=175) & (true<=200), 3, true) # Left Kidney
            true = np.where((true>=240) & (true<=255), 4, true) # Spl
            # Mahesh : Q. Is this the right way to save the numpy array as the nifty label? >> works now after taking tanspose.
            seg = sitk.GetImageFromArray(np.moveaxis(true, [0, 1, 2], [-1, -2, -3]))
            output_file = fnames[0] + ".nii.gz"
            sitk.WriteImage(seg, output_file)
            
            
            pred = np.where((pred>=50) & (pred<=70), 1, pred) # Liver
            pred = np.where((pred>=110) & (pred<=135), 2, pred) # Right Kidney
            pred = np.where((pred>=175) & (pred<=200), 3, pred) # Left Kidney
            pred = np.where((pred>=240) & (pred<=255), 4, pred) # Spl
            # Mahesh : Q. Is this the right way to save the numpy array as the nifty label? >> works now after taking tanspose.
            seg = sitk.GetImageFromArray(np.moveaxis(pred, [0, 1, 2], [-1, -2, -3]))
            output_file = fnames[1] + ".nii.gz"
            sitk.WriteImage(seg, output_file)

        else:
            assert(fnames[0].split('.')[-2] + fnames[0].split('.')[-1] == 'niigz')
            assert(fnames[1].split('.')[-2] + fnames[1].split('.')[-1] == 'niigz')
            pred = pred.detach().cpu()[0, 0, ...].permute(2, 1, 0).numpy() # Mahesh : Should this resize be (1, 2, 0)?
            true = true.detach().cpu()[0, 0, ...].permute(2, 1, 0).numpy()
            if flow is not None:
                assert(fnames[2].split('.')[-2] + fnames[2].split('.')[-1] == 'niigz')
                assert(len(flow.shape)==5)
                flow = flow.detach().cpu()[0, ...].permute(3, 2, 1, 0).numpy() # Mahesh : Should this resize be (1, 2, 3, 0)?
                flow = sitk.GetImageFromArray(flow)
                sitk.WriteImage(flow, fnames[2])
                
            true = sitk.GetImageFromArray(true)
            pred = sitk.GetImageFromArray(pred)
            sitk.WriteImage(true, fnames[0])
            sitk.WriteImage(pred, fnames[1])

def save_rgbflow(flow, slice_num):
    """saved rgb flow. flow should have dimensions (C, H, W, D).
    Args:
        flow (_type_): deformation field flow
        slice_num (_type_): slice number in the depth
    """
    flow = flow[:, :, :, slice_num]
    flow_rgb = np.zeros((flow.shape[1], flow.shape[2], 3))
    for c in range(3):
        flow_rgb[..., c] = flow[c, :, :]
    lower = np.percentile(flow_rgb, 2)
    upper = np.percentile(flow_rgb, 98)
    flow_rgb[flow_rgb < lower] = lower
    flow_rgb[flow_rgb > upper] = upper
    flow_rgb = (((flow_rgb - flow_rgb.min()) / (flow_rgb.max() - flow_rgb.min())))
    plt.figure()
    plt.imshow(flow_rgb, vmin=0, vmax=1)
    plt.axis('off')
    plt.show()

def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

def load_dict(savepath, model, dismiss_keywords=None): #dismiss_keywords is a list
    pth = torch.load(savepath)
    is_data_parallel = isinstance(model, torch.nn.DataParallel)#model
    new_pth = {}
    for k, v in pth.items():
        if dismiss_keywords is not None:
            exist = sum([dismiss_keywords[i] in k for i in range(len(dismiss_keywords))])
            if exist:
                continue
        if 'module' in k:
            if is_data_parallel: # saved multi-gpu, current multi-gpu
                new_pth[k] = v
            else: # saved multi-gpu, current 1-gpu 
                new_pth[k.replace('module.', '')] = v
        else: 
            if is_data_parallel: # saved 1-gpu, current multi-gpu
                new_pth['module.'+k] = v
            else: # saved 1-gpu, current 1-gpu
                new_pth[k] = v
    # import ipdb; ipdb.set_trace()
    m, u = model.load_state_dict(new_pth, strict=False)
    if m:
        logging.info('Missing: '+' '.join(m))
    if u:
        logging.info('Unexpected: '+' '.join(u))
    return

class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count