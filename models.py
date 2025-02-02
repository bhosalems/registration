import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from losses import *
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('error', UserWarning)
import os
from PIL import Image
from pathlib import Path
import imageio
import SimpleITK as sitk
import nibabel as nib
from utils import *
# from TransMorph_affine import config_affine. as CONFIGS


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        zero = torch.cat([torch.eye(3), torch.zeros([3,1])], 1)[None]
        b = nnf.affine_grid(zero, [1, 1, *size], align_corners=False)
        self.meshgrid = nn.Parameter(nnf.affine_grid(zero, [1, 1, *size], align_corners=False), requires_grad=False) # Apparently affine grid calculates 
        # the sampling grid, i.e. it applies theta transform to get the sampling points.
        self.mode = mode

    def forward(self, src, flow):
        flow = flow.permute(0, 2, 3, 4, 1)
        new_locs = self.meshgrid + flow # at each iteration it refines the flow ?
        self.new_locs = new_locs
        return nnf.grid_sample(src, new_locs, mode=self.mode, align_corners=False) # Grid sampling actually does 
        # sampling from the flow and gives you the output 

class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """
    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 3
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out

class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """
    def __init__(self, dim=3, enc_nf=[16, 32, 32, 32], dec_nf= [32, 32, 32, 32, 32, 16, 16], droprate=0, input_ch=2):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding 
                            layers
        """
        super(unet_core, self).__init__()

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = input_ch if i == 0 else enc_nf[i-1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))
            # Mahesh : Q. Why the drouput is zero here and in the decoder, the dropout >> Not required here, as original voxelmorph doesnt have it. 
            self.enc.append(nn.Dropout(0))
        # self.drop = nn.Dropout(droprate)

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(nn.Dropout(0))
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(nn.Dropout(0))
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(nn.Dropout(0))
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(nn.Dropout(0))
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5
        self.dec.append(conv_block(dim, dec_nf[4] + input_ch, dec_nf[5], 1)) # 6
        

        self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6]) 
 
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        y = x
        for i in range(len(self.enc)):
            layer = self.enc[i]
            y = layer(y)
            # logging.info(y.size())
            # import ipdb; ipdb.set_trace()
            if i%2==0:
                x_enc.append(y)
            # logging.info(layer)

        # Three conv + upsample + concatenate series
        # y = x_enc[-1]
        # import ipdb; ipdb.set_trace()
        # y=self.drop(y)
        concat_cnt = 0
        for i in range(len(self.dec)-1):
            layer = self.dec[i]
            y = layer(y)
            concat = (i%2==0) and (concat_cnt<3)
            if concat:
                y = self.upsample(y)
                # logging.info(i, y.size(), x_enc[-(i+2)].size())
                y = torch.cat([y, x_enc[-(concat_cnt+2)]], dim=1) #(-2,-3,-4)
                concat_cnt += 1
            # Two convs at full_size/2 res
            # y = layer(y)
            # y = self.dec[8](y)
            # y = self.dec[9](y)

        # Upsample to full res, concatenate and conv
        y = self.upsample(y)
        y = torch.cat([y, x_enc[0]], dim=1)
        # y = self.dec[-2](y)
        y = self.dec[-1](y)
        # Extra conv for vm2
        y = self.vm2_conv(y)

        return y


class RegNet(nn.Module):
    def __init__(self, size, dim=3, winsize=7, enc_nf=[16, 32, 32, 32], dec_nf= [32, 32, 32, 32, 32, 16, 
                 16], n_class=29):
        super(RegNet, self).__init__()
        self.unet = unet_core(dim=dim, enc_nf = enc_nf, dec_nf = dec_nf)
        if(dim == 3):
            conv_fn = nn.Conv3d
        else:
            conv_fn = nn.Conv2d
        self.conv = conv_fn(dec_nf[-1], dim, kernel_size = 3, stride = 1, padding = 1)
        self.spatial_transformer_network = SpatialTransformer(size)

        self.winsize = winsize
        self.n_class = n_class
        
        # self.affine_config = config = TransMorph_affine.CONFIGS['TransMorph-Affine']
        # self.afiine_infer  = TransMorph_affine.ApplyAffine()
        # self.affine_model = TransMorph_affine.SwinAffine(config)
        
        #feat
        # self.feat_conv= conv_fn(dec_nf[-1]+num_classes, num_classes, kernel_size = 3, stride = 1, padding = 1)
        # self.softmax = nn.Softmax(dim=1)
        # self.drop = nn.Dropout(droprate)

    def eval_dice(self, fixed_label, moving_label, flow, fix_nopad=None, seg_fname=None, save_nii=False):
        warped_seg = self.spatial_transformer_network(moving_label, flow)
        if fix_nopad is not None:
            # Mahesh : NOTE You dont need same dimensions for of warped_seg and fix_nopad for elementwise product.
            warped_seg = fix_nopad * warped_seg
            moving_label = fix_nopad * moving_label
        # Mahesh : Q. Shouldn't there be argmax() here instead of max()? >> No it actually is taking indices by taking torch.max(...)[1].
        # Mahesh : Q. Not sure if taking max here is really required ? Our warped seg just gives us the single value or score, so max will always be zero here. 
        # Is this the error we are experiencing? >> No, originally when we calculate the max() we are passing the one hot encoded labels, so the output
        # from the spatial transformer is not a single value but the a vector.
        warplabel = torch.max(warped_seg.detach(),dim=1)[1]
        # warplabel = warped_seg.squeeze(0)
        # torch.save(warplabel, 'warplabel2.pt')       
        warpseg = torch.nn.functional.one_hot(warplabel.long(), num_classes=self.n_class).float().permute(0, 4, 1, 2, 3)
        dice = dice_onehot(warpseg[:, 1:, :, :, :].detach(), fixed_label[:, 1:, :, :, :].detach())#disregard background
        # Code to debug why sometimes dice is all zero, it appears that the flow is quite bad.
        # if dice == 0:
        #     fnames = []
        #     save_nii = True
        #     grid_flow = None
        #     grid = mk_grid_img((400, 400, 64, 1, 3), 16, 1) # Mahesh : Somehow flow.shape doesnt work because 
        #     # max and min we get in flow_asrgb() both get value of 1. To overcome, passing hardcoded temporarily,
        #     # the axis along which max() and min() is taken will fix the issue.
        #     grid = np.moveaxis(grid, [3, 4], [-5, -4])
        #     grid = torch.tensor(grid, device=flow.device, dtype=flow.dtype)
        #     grid_flow = self.spatial_transformer_network(grid, flow)
        #     grid_flow = grid_flow.squeeze(0)
        #     if grid_flow is not None:
        #         fnames.append(seg_fname+"grid_flow.png")
        #     flow_as_rgb(grid_flow.detach().cpu().numpy(), 8, fname=fnames[-1])
        if save_nii:
            fnames = []
            fnames.append(seg_fname+"true_seg")
            fnames.append(seg_fname+"pred_seg")
            tensor2nii(pred=warpseg, true=fixed_label, fnames=fnames, mode='seg', one_hot=True)
        seg_fname = None # Temporary
        if seg_fname is not None:
            self.seg_imgs(warpseg, fixed_label, seg_fname, one_hot=True)
        return dice

        
    def dice_val_VOI(self, y_pred, y_true, dice_labels, fix_nopad, seg_fname=None): 
        # Mahesh - Only checks the segmentation DICE of below lables, not all.
        if fix_nopad is not None:
            y_pred = fix_nopad * y_pred
        pred = y_pred.detach().cpu().numpy()[0, 0, ...]
        true = y_true.detach().cpu().numpy()[0, 0, ...]
        DSCs = np.zeros((len(dice_labels)-1, 1)) # ignore the background
        idx = 0
        for i in dice_labels:
            # Ignore the background
            if i == 0:
                continue
            pred_i = pred == i
            true_i = true == i
            intersection = pred_i * true_i
            intersection = np.sum(intersection)
            union = np.sum(pred_i) + np.sum(true_i)
            # print(intersection, union)
            dsc = (2.*intersection) / (union + 1e-5)
            DSCs[idx] = dsc
            idx += 1
        mean_DSC = np.mean(DSCs)
        seg_fname = None # If you need to visualize the grad and images by slices.. but bw gradgflow visualization is better.
        if seg_fname is not None:
            self.seg_imgs(y_pred, y_true, seg_fname+"iou"+str(mean_DSC*100)+".png")
        return mean_DSC

            
    def seg_imgs(self, y_pred, y_true, fname, one_hot=False):
        if one_hot:
           y_pred = torch.max(y_pred.detach(), dim=1)[1]
           y_pred = y_pred.unsqueeze(0)
           y_true = torch.max(y_true.detach(), dim=1)[1]
           y_true = y_true.unsqueeze(0)
        fig_rows = 4
        fig_cols = 4
        n_subplots = fig_rows * fig_cols
        n_slice = y_true.shape[4]
        step_size = n_slice // n_subplots
        plot_range = n_subplots * step_size
        start_stop = int((n_slice - plot_range) / 2)
        fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])
        y_true = np.array(y_true.detach().cpu().numpy()[0, 0, ...], dtype='float64')
        y_pred = np.array(y_pred.detach().cpu().numpy()[0, 0, ...], dtype='float64')
        y_true[ y_true==0 ] = np.nan
        for idx, img in enumerate(range(start_stop, plot_range, step_size)):
            axs.flat[idx].imshow(y_true[:, :, img])
            axs.flat[idx].imshow(y_pred[:, :, img], alpha=0.8)
            axs.flat[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig(fname)
        plt.cla()
        plt.close()
        
    # By deafult we give dice labels of the BraTS datatset. If you are not one-hot encoding the dataset, you have to
    # use the labels for calculating the DICE scores.
    def forward(self, fix, moving, fix_label, moving_label, fix_nopad=None, rtloss=True, eval=True, dice_labels=[0, 1, 2, 3, 4], 
                seg_fname = None):
        
        # moving = moving.unsqueeze(-5)
        # fix = fix.unsqueeze(-5)
        
        # Do affine alignment first
        # x = torch.cat([moving, fix], dim = 1)
        # ct_aff, mat, inv_mats = self.affine_model(x)
        # phan = fix
        # affine_sim_loss, affine_sim_mask = ncc_loss(ct_aff, phan, reduce_mean=False, winsize=self.winsize)
        # if affine_sim_mask.any():
        #     affine_sim_loss = affine_sim_loss[affine_sim_mask].mean()
        # else:
        #     affine_sim_loss = torch.Tensor(np.array([0]))[0]
        
        # x = torch.cat([ct_aff, fix], dim = 1)
        x = torch.cat([moving, fix], dim=1)
        unet_out = self.unet(x)
        save_nii = False
        flow = self.conv(unet_out)
         
        if rtloss:
            warp = self.spatial_transformer_network(moving, flow)
            if fix_nopad is not None:
                fix = fix * fix_nopad
                warp = warp * fix_nopad
            sim_loss, sim_mask = ncc_loss(warp, fix, reduce_mean=False, winsize=self.winsize) #[0,1]
            if sim_mask.any():
                sim_loss = sim_loss[sim_mask].mean()
            else:
                sim_loss = torch.Tensor(np.array([0]))[0]
            # sim_loss = vxvm_ncc(y_true=fix, y_pred=warp, win=[self.winsize, self.winsize, self.winsize])
            # tmp_sim, tmp_mask = ncc_loss(warp, fix, reduce_mean=False, winsize=self.winsize) #[0,1]
            grad_loss = gradient_loss(flow, keepdim=False).mean()
            if fix_nopad is not None:
                mask = fix_nopad.bool()
                sim_mask = sim_mask*mask
            sloss = sim_loss
            
            # Mahesh : If similarity loss is greater than 65%, we will convert tensors to nifti for visualization.
            if sloss*-1 >= 0.80:
                fnames = []
                fnames.append(seg_fname+"true.nii.gz")
                fnames.append(seg_fname+"warp.nii.gz")
                fnames.append(seg_fname+"flow.nii.gz")
                grid_flow = None
                # grid = mk_grid_img((400, 400, 64, 1, 3), 16, 1) 
                grid = mk_grid_img((256, 256, 64, 1, 3), 16, 1) # Mahesh : #TODO Somehow flow.shape doesnt work because 
                # max and min we get in flow_asrgb() both get value of 1. To overcome, passing hardcoded temporarily,
                # the axis along which max() and min() is taken will fix the issue.
                grid = np.moveaxis(grid, [3, 4], [-5, -4])
                grid = torch.tensor(grid, device=flow.device, dtype=flow.dtype)
                grid_flow = self.spatial_transformer_network(grid, flow)
                grid_flow = grid_flow.squeeze(0)
                if grid_flow is not None:
                    fnames.append(seg_fname+"grid_flow.png")
                flow_as_rgb(grid_flow.detach().cpu().numpy(), 8, fname=fnames[-1])
                tensor2nii(warp, fix, fnames, mode='volume', one_hot=True, flow=flow, grid_flow=None)
                save_nii = True
            if eval:
                dice = self.eval_dice(fix_label, moving_label, flow, fix_nopad, seg_fname=seg_fname, save_nii=save_nii)
                # warped_seg = self.spatial_transformer_network(moving_label, flow)
                # warped_seg = torch.max(warped_seg.detach(),dim=1)[1]
                # dice  = self.dice_val_VOI(warped_seg, fix_label, dice_labels, fix_nopad, seg_fname)
                # logging.info(f'eval_dice : {e_dice} dice : {dice}')
                return sloss, grad_loss, dice
            else:
                return sloss, grad_loss
        else:
            if eval:
                dice = self.eval_dice(fix_label, moving_label, flow, fix_nopad, seg_fname=seg_fname, save_nii=save_nii)
                # warped_seg = self.spatial_transformer_network(moving_label, flow)
                # warped_seg = torch.max(warped_seg.detach(),dim=1)[1]
                # dice = self.dice_val_VOI(warped_seg, fix_label, dice_labels, fix_nopad, seg_fname)
                # logging.info(f'eval_dice : {e_dice} dice : {dice}')
                return dice
            else:
                return flow


class RegUncertNet(RegNet):
    def __init__(self, size, dim=3, winsize=7,
                enc_nf=[16, 32, 32, 32], dec_nf= [32, 32, 32, 32, 32, 16, 16], n_class=29, grad_weight=0):
        super(RegNet, self).__init__()
        self.unet = unet_core(dim=dim, enc_nf = enc_nf, dec_nf = dec_nf)
        if(dim == 3):
            conv_fn = nn.Conv3d
        else:
            conv_fn = nn.Conv2d
        self.conv = conv_fn(dec_nf[-1], dim, kernel_size = 3, stride = 1, padding = 1)
        # self.spatial_transformer_network = SpatialTransformer(size)
        # Temporarily change the size for Brats dataset.
        self.spatial_transformer_network = SpatialTransformer([240, 240, 144])
        self.winsize = winsize
        self.n_class = n_class
        #var
        self.conv_var = conv_fn(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv_var_2 = conv_fn(32, n_class, kernel_size = 3, stride = 1, padding = 1)
        self.softplus = nn.Softplus()
        self.grad_weight = grad_weight
        #feat
        # self.feat = feat
        # self.feat_conv= conv_fn(dec_nf[-1]+num_classes, num_classes, kernel_size = 3, stride = 1, padding = 1)
        # self.softmax = nn.Softmax(dim=1)
        # self.drop = nn.Dropout(droprate)
    
    def forward(self, fix, moving, fix_label, moving_label, gtflow=None, rtloss=True, eval=True):
        x = torch.cat([moving,fix], dim = 1)
        unet_out = self.unet(x)
        flow = self.conv(unet_out)
        out = self.conv_var(unet_out)
        out = self.conv_var_2(out)
        flow_var = self.softplus(out)

        warp = self.spatial_transformer_network(moving, flow)
        #
        if rtloss:
            #var
            
            dis = torch.nn.L1Loss(reduction='none')(flow, gtflow).mean(dim=1, keepdim=True)
            loss = (dis/flow_var+ flow_var.log()).mean()
            if self.grad_weight>0:
                loss += gradient_loss(flow, keepdim=False).mean()
            # sim_loss, sim_mask = ncc_loss(warp,fix, reduce_mean=False, winsize=self.winsize) #[0,1]
            # grad_loss = gradient_loss(flow, keepdim=False).mean()
            # sloss = sim_loss[sim_mask].mean()
            if eval:
                dice = self.eval_dice(fix_label, moving_label, flow)
                return loss, dice
            else:
                return loss
        else:
            if eval:
                dice = self.eval_dice(fix_label, moving_label, flow)
                return dice
            else:
                warped_seg= self.spatial_transformer_network(moving_label, flow)
                warplabel = torch.max(warped_seg.detach(),dim=1)[1]
                warpseg = torch.nn.functional.one_hot(warplabel.long(), num_classes=self.n_class).float().permute(0,4,1,2,3)
                return warpseg, flow, flow_var