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
import matplotlib.pyplot as plt

# Running this function separatellt once the flow is saved.
def flow_as_rgb(org_flow, num_slices, fname='foo.png'):
    total_slices = org_flow.shape[3]
    slice_interval = total_slices//num_slices
    
    if num_slices % 4 == 0:
        f, axarr = plt.subplots((num_slices)//4, 4, sharey=True, figsize=(15, 15))
    else:
        f, axarr = plt.subplots((num_slices//4) + 1, 4, sharey=True, figsize=(15, 15))
    i = 0 
    axarr = axarr.flatten()
    
    slices = list(range(0, total_slices, slice_interval))
    for ax in axarr:
        flow = org_flow[:, :, :, slices[i]]
        flow_rgb = np.zeros((flow.shape[1], flow.shape[2], 3))
        for c in range(3):
            flow_rgb[..., c] = flow[c, :, :]
        lower = np.percentile(flow_rgb, 2)
        upper = np.percentile(flow_rgb, 98)
        flow_rgb[flow_rgb < lower] = lower
        flow_rgb[flow_rgb > upper] = upper
        flow_rgb = (((flow_rgb - flow_rgb.min()) / (flow_rgb.max() - flow_rgb.min())))
        # plt.figure()
        ax.imshow(flow_rgb, vmin=0, vmax=1)
        ax.axis('off')
        i+=1
    # plt.show()
    plt.savefig(fname)

def ndgrid(*args, **kwargs):
    """
    Disclaimer: This code is taken directly from the scitools package [1]
    Since at the time of writing scitools predominantly requires python 2.7 while we work with 3.5+
    To avoid issues, we copy the quick code here.
    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    Ref : https://github.com/adalca/pystrum/blob/0e7a47e5cc62725dfadc728351b89162defca696/pystrum/pynd/ndutils.py#L208
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args, **kwargs)

def bw_grid(vol_shape, spacing, thickness=1):
    """
    draw a black and white ND grid.
    Ref : https://github.com/adalca/pystrum/blob/0e7a47e5cc62725dfadc728351b89162defca696/pystrum/pynd/ndutils.py#L208
    Parameters
    ----------
        vol_shape: expected volume size
        spacing: scalar or list the same size as vol_shape
    Returns
    -------
        grid_vol: a volume the size of vol_shape with white lines on black background
    """

    # check inputs
    if not isinstance(spacing, (list, tuple)):
        spacing = [spacing] * len(vol_shape)
    spacing = [f + 1 for f in spacing]
    assert len(vol_shape) == len(spacing)

    # go through axes
    grid_image = np.zeros(vol_shape)
    for d, v in enumerate(vol_shape):
        rng = [np.arange(0, f) for f in vol_shape]
        for t in range(thickness):
            rng[d] = np.append(np.arange(0 + t, v, spacing[d]), -1)
            grid_image[ndgrid(*rng)] = 1

    return grid_image
def mk_grid_img(shp, grid_step, line_thickness=1):
    grid_img = np.zeros(shp)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[0], grid_step):
        grid_img[i+line_thickness-1, :, :] = 1
    return grid_img
# def mk_grid_img(shp, grid_step, line_thickness=1):
#     """Returns black and white grid of the given shape with given thickness and takes steps in the lines.
#     Ref:https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/issues/14 
#     Args:
#         shp (_type_): shape of te volume.
#         grid_step (_type_): number of steps.
#         line_thickness (int, optional): thickness of line. Defaults to 1.

#     Returns:
#         _type_: frid of given shape.
#     """
#     grid_img = np.zeros(shp)
#     for j in range(0, grid_img.shape[1], grid_step):
#         grid_img[:, j+line_thickness-1, :] = 1
#     for i in range(0, grid_img.shape[0], grid_step):
#         grid_img[i+line_thickness-1, :, :] = 1
#     return grid_img

def tensor2nii(pred, true, fnames, mode, one_hot=True, flow=None, grid_flow=None):
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
            
            if grid_flow is not None:
                assert(fnames[3].split('.')[-2] + fnames[3].split('.')[-1] == 'niigz')
                assert(len(grid_flow.shape)==5)
                grid_flow = grid_flow.detach().cpu()[0, ...].permute(3, 2, 1, 0).numpy() # Mahesh : Should this resize be (1, 2, 3, 0)?
                grid_flow = sitk.GetImageFromArray(grid_flow)
                sitk.WriteImage(grid_flow, fnames[3])
            
            true = sitk.GetImageFromArray(true)
            pred = sitk.GetImageFromArray(pred)
            sitk.WriteImage(true, fnames[0])
            sitk.WriteImage(pred, fnames[1])

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