import torch
import torch.nn.functional as F
import numpy as np
import math
import nibabel

def dice_onehot(vol1, vol2):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''

    # dicem = np.zeros(len(labels))labels=[1,2]
    # for idx, lab in enumerate(labels):
    #     # vol1l = vol1 == lab
    #     # vol2l = vol2 == lab
    #     vol1l = vol1[:,lab,:,:,:]
    #     vol2l = vol2[:,lab,:,:,:]
    #     top = 2 * np.sum(np.logical_and(vol1l, vol2l))
    #     bottom = np.sum(vol1l) + np.sum(vol2l)
    #     bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
    #     dicem[idx] = top / bottom

    # return dicem
    numerator = 2*vol1*vol2
    denominator = vol1 + vol2
    division = (numerator.sum(dim=[0,2,3,4]) / denominator.sum(dim=[0,2,3,4]))
    # import ipdb; ipdb.set_trace()
    return division.mean()

    
def D(pred, sg, num_classes=3):
    
    #sg_onehot = F.one_hot(sg.long(), num_classes=3).float().permute(0,4,1,2,3)
    numerator = 2*pred*sg
    denominator = pred + sg
    division = - (numerator.sum(dim=[2,3,4]) / denominator.sum(dim=[2,3,4]))
    division = division.mean(dim=1)
    #import ipdb; ipdb.set_trace()

    return division.mean()


def tversky_loss(s_0, sg_0, s_1, sg_1):
    return D(s_0, sg_0), D(s_1, sg_1)


def gradient_loss(s, penalty='l2', keepdim=False):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :]) 
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :]) 
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1]) 

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
    # import ipdb; ipdb.set_trace()
    if keepdim:
        x_size, y_size, z_size = torch.tensor(s.shape), torch.tensor(s.shape), torch.tensor(s.shape)
        y_size[2] = 1
        x_size[3] = 1
        z_size[4] = 1
        zero = torch.zeros(torch.Size(y_size), device = dy.device)
        dy = torch.cat((zero,dy), dim=2)
        zero = torch.zeros(torch.Size(x_size), device = dx.device)
        dx = torch.cat((zero,dx), dim=3)
        zero = torch.zeros(torch.Size(z_size), device = dz.device)
        dz = torch.cat((zero,dz), dim=4)
        d = torch.mean(dx+dy+dz, dim=1, keepdim=True)
    else:
        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def mse_loss(x, y):
    return torch.mean( (x - y) ** 2 ) 

def vxvm_ncc(y_true, y_pred, win):
    Ii = y_true
    Ji = y_pred

    # get dimension of volume
    # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(Ii.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [7] * ndims if win is None else win
    # print(win)
    
    # compute filters
    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    # conv_fn = getattr(F, 'conv%dd' % ndims)

    # print(conv_fn)
    # compute CC squares
    I2 = Ii * Ii
    J2 = Ji * Ji
    IJ = Ii * Ji

    # print(stride)
    I_sum = F.conv3d(Ii, sum_filt, stride=stride, padding=padding)
    J_sum = F.conv3d(Ji, sum_filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -torch.mean(cc)

def ncc_loss(I, J, winsize=5, reduce_mean=True):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    win = [winsize] * ndims
    
    conv_fn = getattr(F, 'conv%dd' % ndims)
    I2 = I*I
    J2 = J*J
    IJ = I*J

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)
    #avoid overflow
    # I = I.type(torch.float64)
    # J = J.type(torch.float64)
    # sum_filt = sum_filt.type(torch.float64)
    # import ipdb; ipdb.set_trace()
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    # torch.save(I, 'I.pt')
    # torch.save(J, 'J.pt')
    # print("I sum:" + str(torch.sum(I)))
    # print("J sum:" + str(torch.sum(J)))
    mask = ((I_var*J_var) == 0) & ((cross*cross) == 0)
    cc = cross*cross / (I_var*J_var + 1e-5)
    # print("CC output" +  str(torch.sum(torch.isnan(cc)).item()))
    # cc = cross*cross / (I_var*J_var)
    # 
    # error_mask = (cc>1.01) or (cc<0)
    # cc[error_mask] = 0
    # if error:
    #     import ipdb; ipdb.set_trace()
    if reduce_mean:
        return -1 * torch.mean(cc[~mask])
    else:
        # cc[mask] = 0
        return -cc, ~mask

def mask_ncc(pred, sg, reduce_mean = True, winsize=5):
    # rtmean = (mask is None) and (reduce_mean)
    ncc = []
    nonzero_mask = []
    for n in range(pred.shape[1]): #class
        clsn, clsm = ncc_loss(pred[:,n, :, :, :].unsqueeze(dim=1), sg[:,n, :, :, :].unsqueeze(dim=1), reduce_mean=False, winsize=winsize)
        # import ipdb; ipdb.set_trace()
        # errmask = (clsn<-1.01) + (clsn>0)
        # clsn[errmask] = 0
        #
        ncc.append(clsn)
        nonzero_mask.append(clsm)
    # 
    ncc = torch.cat(ncc,dim=1)#.squeeze()
    nonzero_mask = torch.cat(nonzero_mask,dim=1)#.squeeze()
    # import ipdb; ipdb.set_trace()
    if reduce_mean:
        # return torch.mean(ncc[nonzero_mask])
        return torch.mean(ncc)
    else:
        return ncc, nonzero_mask
        # if mask is None:
        #     # return ncc, nonzero_mask
        #     return torch.mean(ncc, dim=1).unsqueeze(dim=1), nonzero_mask
        # else:
        #     mask = mask*nonzero_mask
        #     return torch.mean(ncc, dim=1).unsqueeze(dim=1), mask



def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

def mutual_information( y_pred, y_true,
                        min_value = 0.0,
                        max_value = 1.0,
                        num_bins = 100, sigma_ratio = 0.5):

    EPSILON = 1e-07

    ### calculate bin centers ###
    x = np.linspace(min_value, max_value, num_bins + 1)
    bin_centers = (x[:-1] + x[1:]) / 2

    ## clip values ##
    y_true = torch.clamp(y_true, min_value, max_value)
    y_pred = torch.clamp(y_pred, min_value, max_value)

    sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

    bin_centers = torch.from_numpy(bin_centers).float()

    preterm = 1.0 / (2 * np.square(sigma))

    #number of voxels
    nb_voxels = np.prod(list(y_true.shape[1:]))

    y_true = torch.reshape(y_true, [-1, nb_voxels, 1])
    y_pred = torch.reshape(y_pred, [-1, nb_voxels, 1])

    vbc = torch.reshape(bin_centers, [1, 1, num_bins])

    I_a = torch.exp(-preterm * ((y_true - vbc)**2))
    I_a /= torch.sum(I_a, -1, keepdims = True)

    I_b = torch.exp(-preterm * ((y_pred - vbc)**2))
    I_b /= torch.sum(I_b, -1, keepdims = True)

    pab = torch.bmm(I_a.permute(0, 2, 1), I_b) ##batch matrix multiplication
    pab /= nb_voxels

    pa = torch.mean(I_a, 1, keepdims = True)
    pb = torch.mean(I_b, 1, keepdims = True)

    papb = torch.bmm(pa.permute(0, 2, 1), pb) + EPSILON
    # mi = torch.sum(torch.sum(pab * torch.log(pab / papb + EPSILON)), 1)
    #import ipdb; ipdb.set_trace()
    #batch*num_bins*num_bins
    mi = torch.sum(pab * torch.log(pab / papb + EPSILON), (1,2)).mean()
    return mi


# def pad(x, shape):
#     s_x = math.floor((shape[0] - x.shape[0])/2)
#     s_y = math.floor((shape[1] - x.shape[1])/2)
#     s_z = math.floor((shape[2] - x.shape[2])/2)
#     new_x = np.zeros(shape)
#     new_x[s_x:s_x+x.shape[0],s_y: s_y + x.shape[1], s_z:s_z + x.shape[2]] = x
#     save_index = [s_x, x.shape[0], s_y, x.shape[1], s_z, x.shape[2]]
#     # nopad = np.zeros_like(new_x)
#     # nopad[s_x:s_x+x.shape[0],s_y: s_y + x.shape[1], s_z:s_z + x.shape[2]] = 1
#     return new_x, save_index

# def normalize(x):
#     mean = np.mean(x)
#     std = np.std(x, ddof=1)
#     #std_arr = np.sqrt(np.abs(x-mean)/x.size)
#     maxp = mean + 6*std
#     minp = mean - 6*std
#     y = np.clip(x, minp, maxp)
#     #import ipdb; ipdb.set_trace()
#     #linear transform to [0,1]
#     z = (y-y.min())/y.max()
#     return z

# def pre_hippo(name, shape = (48,64,48), resample=[1,1,1], isimg=False):
#     image = np.array(nibabel.load(name).get_fdata())
#     if isimg:
#         image = normalize(image)
#     image, nopad = pad(image, shape)
#     return image, nopad


if __name__ == '__main__':
    # data = torch.tensor([0.5, 0.3, 0.2, 0.2, 0.6])
    # gt = torch.tensor([1, 0, 0, 0, 1])
    # t_loss = get()
    path = '/home/xuangong/data/hippocampus/all/data/'
    import os
    #import dataloader
    files = os.listdir(path)
    for f in files:
        if f.endswith('.nii.gz'):
            data, idx = pre_hippo(os.path.join(path, f))
            x1, x2, y1, y2, z1, z2 = idx[0], idx[0]+idx[1], \
                idx[2], idx[2]+idx[3], idx[4], idx[4]+idx[5]
            clip_data= data[x1:x2,y1:y2,z1:z2]
            #clip_data= data
            data = torch.tensor(clip_data[None, None]).float().cuda()
            data2 = torch.tensor(data[None, None]).float().cuda()
            print(mutual_information(data2.detach().cpu(),data2.detach().cpu()), \
                mutual_information(data.detach().cpu(),data.detach().cpu()))
            #import ipdb; ipdb.set_trace()

# def dicescore(vol1, vol2, labels=None, nargout=1):
#     '''
#     Dice [1] volume overlap metric

#     The default is to *not* return a measure for the background layer (label = 0)

#     [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
#     Ecology 26.3 (1945): 297-302.+

#     Parameters
#     ----------
#     vol1 : nd array. The first volume (e.g. predicted volume)
#     vol2 : nd array. The second volume (e.g. "true" volume)
#     labels : optional vector of labels on which to compute Dice.
#         If this is not provided, Dice is computed on all non-background (non-0) labels
#     nargout : optional control of output arguments. if 1, output Dice measure(s).
#         if 2, output tuple of (Dice, labels)

#     Output
#     ------
#     if nargout == 1 : dice : vector of dice measures for each labels
#     if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
#         dice was computed
#     '''
#     if labels is None:
#         labels = np.unique(np.concatenate((vol1, vol2)))
#         labels = np.delete(labels, np.where(labels == 0))  # remove background

#     dicem = np.zeros(len(labels))
#     for idx, lab in enumerate(labels):
#         vol1l = vol1 == lab
#         vol2l = vol2 == lab
#         top = 2 * np.sum(np.logical_and(vol1l, vol2l))
#         bottom = np.sum(vol1l) + np.sum(vol2l)
#         bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
#         dicem[idx] = top / bottom

#     if nargout == 1:
#         return dicem
#     else:
#         return (dicem, labels)