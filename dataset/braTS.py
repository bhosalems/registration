import os
from sklearn import neighbors
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import math
import numpy as np
import random
import warnings
import nibabel as nib
from nilearn.image import new_img_like, reorder_img, resample_to_img, smooth_img
from nilearn.image.image import check_niimg
from collections.abc import Iterable
from torch.utils.data.dataloader import default_collate

def normalization_name_to_function(normalization_name):
    if type(normalization_name) == list:
        return partial(normalize_data_with_multiple_functions, normalization_names=normalization_name)
    elif normalization_name == "zero_mean":
        return zero_mean_normalize_image_data
    elif normalization_name == "foreground_zero_mean":
        return foreground_zero_mean_normalize_image_data
    elif normalization_name == "zero_floor":
        return zero_floor_normalize_image_data
    elif normalization_name == "zero_one_window":
        return zero_one_window
    elif normalization_name == "mask":
        return mask
    elif normalization_name is not None:
        try:
            return getattr(normalize, normalization_name)
        except AttributeError:
            raise NotImplementedError(normalization_name + " normalization is not available.")
    else:
        return lambda x, **kwargs: x

def center_crop(x,size):
    ori_size=x.shape
    pad = [int((ori_size[i]-size[i])/2) for i in [0,1,2]]
    y = x[pad[0]:pad[0]+size[0], pad[1]:pad[1]+size[1], pad[2]:pad[2]+size[2]]
    return y

def image_slices_to_affine(image, slices):
    affine = image.affine

    linear_part = affine[:3, :3]
    old_origin = affine[:3, 3]
    new_origin_voxel = np.array([s.start for s in slices])
    new_origin = old_origin + linear_part.dot(new_origin_voxel)

    new_affine = np.eye(4)
    new_affine[:3, :3] = linear_part
    new_affine[:3, 3] = new_origin
    return new_affine

def crop_img(img, rtol=1e-8, copy=True, return_slices=False, pad=True, percentile=None, return_affine=False):
    """Crops img as much as possible
    Will crop img, removing as many zero entries as possible
    without touching non-zero entries. Will leave one voxel of
    zero padding around the obtained non-zero area in order to
    avoid sampling issues later on.
    Parameters
    ----------
    img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        img to be cropped.
    rtol: float
        relative tolerance (with respect to maximal absolute
        value of the image), under which values are considered
        negligeable and thus croppable.
    copy: boolean
        Specifies whether cropped data is copied or not.
    return_slices: boolean
        If True, the slices that define the cropped image will be returned.
    pad: boolean or integer
        If True, an extra slice in each direction will be added to the image. If integer > 0 then the pad width will
        be set to that integer.
    percentile: integer or None
        If not None, then the image will be crop out slices below the given percentile
    Returns
    -------
    cropped_img: image
        Cropped version of the input image
    """

    img = check_niimg(img)
    data = img.get_fdata()
    if percentile is not None:
        passes_threshold = data > np.percentile(data, percentile)
    else:
        infinity_norm = max(-data.min(), data.max())
        passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                         data > rtol * infinity_norm)

    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    if int(pad) > 0:
        pad_width = int(pad)
        # pad with one voxel to avoid resampling problems
        start = np.maximum(start - pad_width, 0)
        end = np.minimum(end + pad_width, data.shape[:3])

    slices = [slice(s, e) for s, e in zip(start, end)]

    if return_slices:
        return slices

    if return_affine:
        return image_slices_to_affine(img, slices), end - start

    return crop_img_to(img, slices, copy=copy)

def decision(probability):
    if not probability or probability <= 0:
        return False
    elif probability >= 1:
        return True
    else:
        return random.random() < probability

def is_diag(x):
    return np.count_nonzero(x - np.diag(np.diagonal(x))) == 0

def assert_affine_is_diagonal(affine):
    if not is_diag(affine[:3, :3]):
        raise NotImplementedError("Hemisphere swapping for non-diagonal affines is not yet implemented.")

def affine_swap_axis(affine, shape, axis=0):
    assert_affine_is_diagonal(affine)
    new_affine = np.copy(affine)
    origin = affine[axis, 3]
    new_affine[axis, 3] = origin + shape[axis] * affine[axis, axis]
    new_affine[axis, axis] = -affine[axis, axis]
    return new_affine

def find_center(affine, shape, ndim=3):
    return np.matmul(affine,
                     list(np.divide(shape[:ndim], 2)) + [1])[:ndim]

def scale_affine(affine, shape, scale, ndim=3):
    """
    This assumes that the shape stays the same.
    :param affine: affine matrix for the image.
    :param shape: current shape of the data. This will remain the same.
    :param scale: iterable with length ndim, int, or float. A scale greater than 1 indicates the image will be zoomed,
    the spacing will get smaller, and the affine window will be smaller as well. A scale of less than 1 indicates
    zooming out with the spacing getting larger and the affine window getting bigger.
    :param ndim: number of dimensions (default is 3).
    :return:
    """
    if not isinstance(scale, Iterable):
        scale = np.ones(ndim) * scale
    else:
        scale = np.asarray(scale)

    # 1. find the image center
    center = find_center(affine, shape, ndim=ndim)

    # 2. translate the affine
    affine = affine.copy()
    origin = affine[:ndim, ndim]
    t = np.diag(np.ones(ndim + 1))
    t[:ndim, ndim] = (center - origin) * (1 - 1 / scale)
    affine = np.matmul(t, affine)

    # 3. scale the affine
    s = np.diag(list(1 / scale) + [1])
    affine = np.matmul(affine, s)
    return affine

def translate_affine(affine, shape, translation_scales, copy=True):
    """
    :param translation_scales: (tuple) Contains x, y, and z translations scales from -1 to 1. 0 is no translation.
    1 is a forward (RAS-wise) translation of the entire image extent for that direction. -1 is a translation in the
    negative direction of the entire image extent. A translation of 1 is impractical for most purposes, though, as it
    moves the image out of the original field of view almost entirely. To perform a random translation, you can
    use numpy.random.normal(loc=0, scale=sigma, size=3) where sigma is the percent of image translation that would be
    randomly translated on average (0.05 for example).
    :return: affine
    """
    if copy:
        affine = np.copy(affine)
    spacing = get_spacing_from_affine(affine)
    extent = np.multiply(shape, spacing)
    translation = np.multiply(translation_scales, extent)
    affine[:3, 3] += translation
    return affine

def augment_affine(affine, shape, augment_scale_std=None, augment_scale_probability=1,
                   flip_left_right_probability=0, augment_translation_std=None, augment_translation_probability=1,
                   flip_front_back_probability=0):
    if augment_scale_std and decision(augment_scale_probability):
        scale = np.random.normal(1, augment_scale_std, 3)
        affine = scale_affine(affine, shape, scale)
    if decision(flip_left_right_probability):  # flips the left and right sides of the image randomly
        affine = affine_swap_axis(affine, shape=shape, axis=0)
    if decision(flip_front_back_probability):
        affine = affine_swap_axis(affine, shape=shape, axis=1)
    if augment_translation_std and decision(augment_translation_probability):
        affine = translate_affine(affine, shape,
                                  translation_scales=np.random.normal(loc=0, scale=augment_translation_std, size=3))
    return affine

def random_blur(image, mean, std):
    """
    mean: mean fwhm in millimeters.
    std: standard deviation of fwhm in millimeters.
    """
    return smooth_img(image, fwhm=np.abs(np.random.normal(mean, std, 3)).tolist())

def add_noise(data, mean=0., sigma_factor=0.1):
    """
    Adds Gaussian noise.
    :param data: input numpy array
    :param mean: mean of the additive noise
    :param sigma_factor: standard deviation of the image will be multiplied by sigma_factor to obtain the standard
    deviation of the additive noise. Assumes standard deviation is the same for all channels.
    :return:
    """
    sigma = np.std(data) * sigma_factor
    noise = np.random.normal(mean, sigma, data.shape)
    return np.add(data, noise)

def augment_image(image, augment_blur_mean=None, augment_blur_std=None, augment_blur_probability=1,
                  additive_noise_std=None, additive_noise_probability=1):
    if not (augment_blur_mean is None or augment_blur_std is None) and decision(augment_blur_probability):
        image = random_blur(image, mean=augment_blur_mean, std=augment_blur_std)
    if additive_noise_std and decision(additive_noise_probability):
        image.dataobj[:] = add_noise(image.dataobj, sigma_factor=additive_noise_std)
    return image

def get_spacing_from_affine(affine):
    RZS = affine[:3, :3]
    return np.sqrt(np.sum(RZS * RZS, axis=0))

def calculate_origin_offset(new_spacing, old_spacing):
    return np.divide(np.subtract(new_spacing, old_spacing)/2, old_spacing)

def set_affine_spacing(affine, spacing):
    scale = np.divide(spacing, get_spacing_from_affine(affine))
    affine_transform = np.diag(np.ones(4))
    np.fill_diagonal(affine_transform, list(scale) + [1])
    return np.matmul(affine, affine_transform)

def adjust_affine_spacing(affine, new_spacing, spacing=None):
    if spacing is None:
        spacing = get_spacing_from_affine(affine)
    offset = calculate_origin_offset(new_spacing, spacing)
    new_affine = np.copy(affine)
    translation_affine = np.diag(np.ones(4))
    translation_affine[:3, 3] = offset
    new_affine = np.matmul(new_affine, translation_affine)
    new_affine = set_affine_spacing(new_affine, new_spacing)
    return new_affine

def resize_affine(affine, shape, target_shape, copy=True):
    if copy:
        affine = np.copy(affine)
    scale = np.divide(shape, target_shape)
    spacing = get_spacing_from_affine(affine)
    target_spacing = np.multiply(spacing, scale)
    affine = adjust_affine_spacing(affine, target_spacing)
    return affine

# Mahesh : This is the function which does the actual augmentation. (resample_input->resample_image->format_feature_image).
def format_feature_image(feature_image, window, crop=False, cropping_kwargs=None, augment_scale_std=None,
                         augment_scale_probability=1, additive_noise_std=None, additive_noise_probability=0,
                         flip_left_right_probability=0, augment_translation_std=None,
                         augment_translation_probability=0, augment_blur_mean=None, augment_blur_std=None,
                         augment_blur_probability=0, flip_front_back_probability=0, reorder=False,
                         interpolation="linear"):
    if reorder:
        feature_image = reorder_img(feature_image, resample=interpolation)
    if crop:
        if cropping_kwargs is None:
            cropping_kwargs = dict()
        affine, shape = crop_img(feature_image, return_affine=True, **cropping_kwargs)
    else:
        affine = feature_image.affine.copy()
        shape = feature_image.shape
    affine = augment_affine(affine, shape,
                            augment_scale_std=augment_scale_std,
                            augment_scale_probability=augment_scale_probability,
                            augment_translation_std=augment_translation_std,
                            augment_translation_probability=augment_translation_probability,
                            flip_left_right_probability=flip_left_right_probability,
                            flip_front_back_probability=flip_front_back_probability)
    feature_image = augment_image(feature_image,
                                  augment_blur_mean=augment_blur_mean,
                                  augment_blur_std=augment_blur_std,
                                  augment_blur_probability=augment_blur_probability,
                                  additive_noise_std=additive_noise_std,
                                  additive_noise_probability=additive_noise_probability)
    affine = resize_affine(affine, shape, window)
    return feature_image, affine

def resample_image(source_image, target_image, interpolation="linear", pad_mode='edge', pad=False):
    if pad:
        source_image = pad_image(source_image, mode=pad_mode)
    return resample_to_img(source_image, target_image, interpolation=interpolation)

def resample(image, target_affine, target_shape, interpolation='linear', pad_mode='edge', pad=False):
    target_data = np.zeros(target_shape, dtype=image.get_data_dtype())
    # TODO Mahesh : What class we get here in original dataset_brats_seg.py
    target_image = image.__class__(target_data, affine=target_affine)
    return resample_image(image, target_image, interpolation=interpolation, pad_mode=pad_mode, pad=pad)

def load_single_image(filename, resample=None, reorder=True):
    image = nib.load(filename)
    if reorder:
        return reorder_img(image, resample=resample)
    return image

def zero_mean_normalize_image_data(data, axis=(0, 1, 2)):
    return np.divide(data - data.mean(axis=axis), data.std(axis=axis))

def nib_load_files(filenames, reorder=False, interpolation="linear"):
    if type(filenames) != list:
        filenames = [filenames]
    return [load_image(filename, reorder=reorder, interpolation=interpolation, force_4d=False)
            for filename in filenames]

def get_nibabel_data(nibabel_image):
    return nibabel_image.get_fdata()

def normalize_image_with_function(image, function, volume_indices=None, **kwargs):
    data = get_nibabel_data(image)
    if volume_indices is not None:
        data[..., volume_indices] = function(data[..., volume_indices], **kwargs)
    else:
        data = function(data, **kwargs)
    return new_img_like(image, data=data, affine=image.affine)

def combine_images(images, axis=0, resample_unequal_affines=False, interpolation="linear"):
    base_image = images[0]
    data = list()
    max_dim = len(base_image.shape)
    for image in images:
        try:
            np.testing.assert_array_equal(image.affine, base_image.affine)
        except AssertionError as error:
            if resample_unequal_affines:
                image = resample_to_img(image, base_image, interpolation=interpolation)
            else:
                raise error
        image_data = image.get_fdata()
        # import ipdb; ipdb.set_trace()
        dim = len(image.shape)
        if dim < max_dim:
            image_data = np.expand_dims(image_data, axis=axis)
        elif dim > max_dim:
            max_dim = max(max_dim, dim)
            data = [np.expand_dims(x, axis=axis) for x in data]
        data.append(image_data)
    if len(data[0].shape) > 3:
        array = np.concatenate(data, axis=axis)
    else:
        # Mahesh : If we ought to use the multiple images later on, we stack it here, these images
        # give you one more etra dimensoion, whcih we dont need.
        if len(images) > 1:
            array = np.stack(data, axis=axis)
        else:
            array = data[0]
    return base_image.__class__(array, base_image.affine)

def load_image(filename, feature_axis=3, resample_unequal_affines=True, interpolation="linear", force_4d=False,
               reorder=False):
    """
    :param feature_axis: axis along which to combine the images, if necessary.
    :param filename: can be either string path to the file or a list of paths.
    :return: image containing either the 1 image in the filename or a combined image based on multiple filenames.
    """

    if type(filename) != list:
        if not force_4d:
            return load_single_image(filename=filename, resample=interpolation, reorder=reorder)
        else:
            filename = [filename]
    
    return combine_images(nib_load_files(filename, reorder=reorder, interpolation=interpolation), axis=feature_axis,
                          resample_unequal_affines=resample_unequal_affines, interpolation=interpolation)

def compile_one_hot_encoding(data, n_labels, labels=None, dtype=np.uint8, return_4d=True):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :param dtype: output type of the array
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    data = np.asarray(data)
    while len(data.shape) < 5:
        data = data[None]
    assert data.shape[1] == 1
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, dtype=dtype)
    for label_index in range(n_labels):
        if labels is not None:
            if type(labels[label_index]) == list:
                # lists of labels will group together multiple labels from the label map into a single one-hot channel.
                for label in labels[label_index]:
                    y[:, label_index][data[:, 0] == label] = 1
            else:
                y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    if return_4d:
        assert y.shape[0] == 1
        y = y[0]
    return y

class BraTSDataset(Dataset):
    def __init__(self, data_path, mod, seg='seg', size=[240, 240, 155], downsample_rate=16, augment=True, feature_index=0, 
                 target_index=1, crop=True, cropping_kwargs=None, interpolation='linear',
                 augment_scale_std=0, augment_scale_probability=1, additive_noise_std=0, additive_noise_probability=1,
                 augment_blur_mean=None, augment_blur_std=None, augment_blur_probability=1,
                 augment_translation_std=None, augment_translation_probability=1, flip_left_right_probability=0,
                 random_permutation_probability=0, flip_front_back_probability=0, resample=None, extract_sub_volumes=False, 
                 normalize=True, normalization="zero_mean",  normalization_args=None, reorder=False, target_interpolation='nearest', 
                 add_contours = False, **kwargs ):
        """ Dataset for https://www.med.upenn.edu/sbia/brats2018/data.html 

        Args:
            data_path (str): Root path of the dataset
            mod (str): Modality of the data
            size (list): image size
        """

        self.size = size
        self.window = np.array(self.size)
        self.datapath = os.path.expanduser(data_path)
        self.mod = mod
        self.augment = augment
        self.seg = seg
        self.labels = [0, 1, 2, 4]
        self.dice_labels = self.labels
        self.downsample_rate = downsample_rate
        self.feature_index = feature_index
        self.target_index = target_index
        self.crop = crop
        self.cropping_kwargs = cropping_kwargs
        self.augment_scale_probability = augment_scale_probability
        self.additive_noise_std = additive_noise_std
        self.additive_noise_probability = additive_noise_probability
        self.interpolation = interpolation
        if resample is not None:
            warnings.warn("'resample' argument is deprecated. Use 'interpolation'.", DeprecationWarning)
        self.augment_scale_std = augment_scale_std
        self.augment_blur_mean = augment_blur_mean
        self.augment_blur_std = augment_blur_std
        self.augment_blur_probability = augment_blur_probability
        self.augment_translation_std = augment_translation_std
        self.augment_translation_probability = augment_translation_probability
        self.flip_left_right_probability = flip_left_right_probability
        self.flip_front_back_probability = flip_front_back_probability
        self.extract_sub_volumes = extract_sub_volumes
        self.normalize = normalize
        self.normalization_func = normalization_name_to_function(normalization)
        if normalization_args is not None:
            self.normalization_kwargs = normalization_args
        else:
            self.normalization_kwargs = dict()
        self.reorder = reorder
        if target_interpolation is None:
            self.target_interpolation = self.interpolation
        else:
            self.target_interpolation = target_interpolation
        self.add_contours = add_contours
        self.random_permutation_probability = random_permutation_probability
        
        # fix
        for fixpath in os.listdir(f'{self.datapath}/fix'):
            for f in os.listdir(f'{self.datapath}/fix/{fixpath}'):
                if self.mod in f:
                    self.fiximg , _= self.preprocess_img(f'{self.datapath}/fix/{fixpath}/{f}')
                    # self.fiximg = self.fiximg[None, ...]
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
        self.imgseg_paths = []
        for subpath in os.listdir(f'{self.datapath}/data'):
            path = os.path.join(f'{self.datapath}/data', subpath)#subject path
            # Mahesh : Relative order of self.mod and self.seg matters which determines the self.feature_index 
            # and self.target_index.
            tmp = []
            for f in sorted(os.listdir(path)):
                if self.mod in f:
                    imgpath = os.path.join(path, f)
                    assert os.path.exists(imgpath)
                    if self.augment:
                         # Get eh filenames as a list of lists of image and corresponding segmentation.
                        tmp.append(imgpath)
                    else:
                        self.imgpath.append(imgpath)
                elif self.seg in f:
                    segpath = os.path.join(path, f)
                    assert os.path.exists(segpath)
                    if self.augment:
                        # Get eh filenames as a list of lists of image and corresponding segmentation.
                        tmp.append(segpath)
                    else:
                        self.segpath.append(segpath)
            if self.augment:
                self.imgseg_paths.append(tmp)

    
    def __getitem__(self, idx):
        if self.augment:
            image, fixed_nopad, seg = self.preprocess(self.imgseg_paths[idx])
        else:
            image, fixed_nopad = self.preprocess_img(self.imgpath[idx])
            seg = self.preprocess_seg(self.segpath[idx])
            # image = image[None, ...]        
        return self.fiximg, self.fixseg, fixed_nopad, image, seg 
    
    def __len__(self):
        if self.augment:
            return len(self.imgseg_paths)
        else:
            return len(self.imgpath)
    
    def zero_pad(self, data, value=0):
        orig_size = data.shape
        c_dim = orig_size[-1]
        pad_sz = abs(c_dim - (math.ceil(c_dim/self.downsample_rate)*self.downsample_rate))
        data = torch.nn.functional.pad(data, (math.floor(pad_sz/2), math.ceil(pad_sz/2)), value=0)
        assert(data.shape[-1]%self.downsample_rate==0)
        return data
    
    
    def normalize_image(self, image):
        if self.normalize:
            return normalize_image_with_function(image, self.normalization_func, **self.normalization_kwargs)
        return image
    
    def load_image(self, filenames, index, force_4d=True, interpolation="linear", sub_volume_indices=None):
        filename = filenames[index]
        # Reordering is done when the image is formatted
        image = load_image(filename, force_4d=force_4d, reorder=False, interpolation=interpolation)
        if sub_volume_indices:
            image = extract_sub_volumes(image, sub_volume_indices)
        return image

    def load_feature_image(self, input_filenames):
        if self.extract_sub_volumes:
            sub_volume_indices = input_filenames[self.feature_sub_volumes_index]
        else:
            sub_volume_indices = None
        return self.load_image(input_filenames, self.feature_index, force_4d=True, interpolation=self.interpolation,
                               sub_volume_indices=sub_volume_indices)    
    
    def format_feature_image(self, input_filenames, return_unmodified=False):
        unmodified_image = self.load_feature_image(input_filenames)
        image, affine = format_feature_image(feature_image=self.normalize_image(unmodified_image),
                                             crop=self.crop,
                                             cropping_kwargs=self.cropping_kwargs,
                                             augment_scale_std=self.augment_scale_std,
                                             augment_scale_probability=self.augment_scale_probability,
                                             window=self.window,
                                             additive_noise_std=None,  # augmented later
                                             augment_blur_mean=None,  # augmented later
                                             augment_blur_std=None,  # augmented later
                                             flip_left_right_probability=self.flip_left_right_probability,
                                             flip_front_back_probability=self.flip_front_back_probability,
                                             augment_translation_std=self.augment_translation_std,
                                             augment_translation_probability=self.augment_translation_probability,
                                             reorder=self.reorder,
                                             interpolation=self.interpolation)
        resampled = resample(image, affine, self.window, interpolation=self.interpolation)
        if return_unmodified:
            return resampled, unmodified_image
        else:
            return resampled

    def load_target_image(self, feature_image, input_filenames):
        if self.target_index is None:
            target_image = copy_image(feature_image)
        else:
            if self.extract_sub_volumes:
                sub_volume_indices = input_filenames[self.target_sub_volumes_index]
            else:
                sub_volume_indices = None
            target_image = self.load_image(input_filenames, self.target_index, force_4d=True,
                                           sub_volume_indices=sub_volume_indices,
                                           interpolation=self.target_interpolation)
        return target_image
    
    def resample_target(self, target_image, feature_image):
        target_image = resample_to_img(target_image, feature_image, interpolation=self.target_interpolation)
        return target_image
    
    def resample_image(self, input_filenames):
        feature_image = self.format_feature_image(input_filenames=input_filenames)
        target_image = self.resample_target(self.load_target_image(feature_image, input_filenames),
                                            feature_image)
        feature_image = augment_image(feature_image,
                                      additive_noise_std=self.additive_noise_std,
                                      additive_noise_probability=self.additive_noise_probability,
                                      augment_blur_mean=self.augment_blur_mean,
                                      augment_blur_std=self.augment_blur_std,
                                      augment_blur_probability=self.augment_blur_probability)
        return feature_image, target_image
    
    def permute_inputs(self, x, y):
        if decision(self.random_permutation_probability):
            x, y = random_permutation_x_y(x, y, channel_axis=self.channel_axis)
        return x, y

    def resample_input(self, input_filenames):
        input_image, target_image = self.resample_image(input_filenames)
        target_data = get_nibabel_data(target_image)
        n_class = len(self.labels)
        seg = np.zeros_like(target_data)
        mapping = {0:0, 1:1, 2:2, 4:3}
        for _,label in enumerate(self.labels):
            newlabel = mapping[label]
            seg[target_data==label] = newlabel
        if self.labels is None:
            self.labels = np.unique(seg)
        
        # import ipdb; ipdb.set_trace()
        # NOTE Mahesh : If you ever use the commented code below, target_data should be replaced by variable seg
        # print(self.labels)
        # print(np.unique(target_data))
        # assert len(target_data.shape) == 4
        # if target_data.shape[3] == 1:
        #     target_data = np.moveaxis(
        #         compile_one_hot_encoding(np.moveaxis(target_data, 3, 0),
        #                                 n_labels=len(self.labels),
        #                                 labels=self.labels,
        #                                 return_4d=True), 0, 3)
        # else:
        #     _target_data = list()
        #     for channel, labels in zip(range(target_data.shape[self.channel_axis]), self.labels):
        #         _target_data.append(np.moveaxis(
        #             compile_one_hot_encoding(np.moveaxis(target_data[..., channel, None], self.channel_axis, 0),
        #                                     n_labels=len(labels),
        #                                     labels=labels,
        #                                     return_4d=True), 0, self.channel_axis))
        #     target_data = np.concatenate(_target_data, axis=self.channel_axis)
        if self.add_contours:
            seg = add_one_hot_encoding_contours(seg)
        return self.permute_inputs(get_nibabel_data(input_image), seg)
    
    def preprocess(self, names):
        """This is the entry point for reading the image and segmentation when augmentation is true

        Args:
            names (_type_): _description_
        """
        img, seg = self.resample_input(names)
        # Mahesh : For now we are not giving all the modalitites together so we just get simgle 3d volume
        # If multiple modalitites are added as given in dataset_brats_seg.py, we need to change couple 
        # of things including removing/changoing this assert. 
        assert(len(img.shape) == 3)
        assert(len(seg.shape) == 3)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        seg = np.ascontiguousarray(seg)
        seg = torch.from_numpy(seg)
        
        # Add the padding if required
        if img.shape[-1]%self.downsample_rate != 0:
            fixed_nopad = torch.ones(img.shape)
            img = self.zero_pad(img)
            fixed_nopad = self.zero_pad(fixed_nopad)
            seg = self.zero_pad(seg)
            self.size = img.shape
        return img, fixed_nopad, seg
                
    def preprocess_img(self, name):
        """Reads and preprocesses the image and calls augmentation code. Clip the image in range 
        [mean + 6*std, mean - 6*std]. at last, normalize the image.

        Args:
            name (str): filepath

        Returns:
            array: Preprocesses image array
        """
        data = np.array(nib.load(name).get_fdata())
        fixed_nopad = None

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
        z = np.ascontiguousarray(z)
        z = torch.from_numpy(z)

        # Add the padding if required
        if z.shape[-1]%self.downsample_rate != 0:
            fixed_nopad = torch.ones(z.shape)
            z = self.zero_pad(z)
            fixed_nopad = self.zero_pad(fixed_nopad)
            self.size = z.shape
        return z, fixed_nopad

    def preprocess_seg(self, name):
        data = np.array(nib.load(name).get_fdata())
        n_class = len(self.labels)
        seg = np.zeros_like(data)
        mapping = {0:0, 1:1, 2:2, 4:3}
        for _,label in enumerate(self.labels):
            newlabel = mapping[label]
            seg[data==label] = newlabel
        
        seg = np.ascontiguousarray(seg)
        seg = torch.from_numpy(seg)
        
        # Add the padding if required
        if seg.shape[-1]%self.downsample_rate != 0:
            seg = self.zero_pad(seg)
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


def braTS_dataloader(root_path, save_path, bsize, mod, seg = "seg", size=[240, 240, 155], data_split=False, n_fix=1, num_workers = 4, augment=True):
    if(data_split):
        train_rootpath = root_path + '_Train'
        validation_rootpath = root_path + '_Validation'
        train_savepath = os.path.join(save_path, "Train")
        validation_savepath = os.path.join(save_path, "Validation")
        datasplit(train_rootpath, train_savepath, n_fix) 
        datasplit(validation_rootpath, validation_savepath, n_fix) 

    tr_path = os.path.join(save_path, "Train")
    ts_path = os.path.join(save_path, "Validation")

    train_dataset = BraTSDataset(tr_path, mod, reorder=True, augment=augment)
    test_dataset =  BraTSDataset(ts_path, mod, reorder=True, augment=augment)

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

    pad_size = [240, 240, 155]
    mod = "flair"
    bsize = 1

    # Need to call data_split only once for train as well as validation at first.
    train_dataloader, test_dataloader = braTS_dataloader(root_path, save_path, bsize, mod, augment=True)

    for _, samples in enumerate(train_dataloader):
        print(samples.shape)
    
    for _, samples in enumerate(test_dataloader):
        print(samples.shape)