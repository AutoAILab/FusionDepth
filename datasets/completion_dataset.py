from __future__ import absolute_import, division, print_function

import cv2
import os
import glob
import random
import numpy as np
import copy
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_paths_and_transform(data_folder, split, val_split, verify=True):
    if split == "train":
        use_d = True
        use_rgb = True
        glob_d = os.path.join(
            data_folder,
            'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            data_folder,
            'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )

        def get_rgb_paths(p):
            ps = p.split('/')
            pnew = '/'.join([data_folder] + ['data_rgb'] + ps[-6:-4] +
                            ps[-2:-1] + ['data'] + ps[-1:])
            return pnew
    elif split == "val":
        use_d = True
        use_rgb = True
        use_g = True
        if val_split == "full":
            glob_d = os.path.join(
                data_folder,
                'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                data_folder,
                'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )
            def get_rgb_paths(p):
                ps = p.split('/')
                pnew = '/'.join(ps[:-7] +
                    ['data_rgb']+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
                return pnew
        elif val_split == "select":
            glob_d = os.path.join(
                data_folder,
                "depth_selection/val_selection_cropped/velodyne_raw/*.png")
            glob_gt = os.path.join(
                data_folder,
                "depth_selection/val_selection_cropped/groundtruth_depth/*.png"
            )
            def get_rgb_paths(p):
                return p.replace("groundtruth_depth","image")
    elif split == "test_completion":
        glob_d = os.path.join(
            data_folder,
            "depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png"
        )
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            data_folder,
            "depth_selection/test_depth_completion_anonymous/image/*.png")
    elif split == "test_prediction":
        glob_d = None
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            data_folder,
            "depth_selection/test_depth_prediction_anonymous/image/*.png")
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d))
        paths_gt = sorted(glob.glob(glob_gt))
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
    else:
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None] * len(paths_rgb)
        print('None paths gt')
        if split == "test_prediction":
            paths_d = [None] * len(
                paths_rgb)  # test_prediction has no sparse depth
        else:
            print('None paths gt')
            paths_d = sorted(glob.glob(glob_d))

    def verify_nearby(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        new_filename = os.path.join(head, '%010d.png' % (number - 1))
        new_filename1 = os.path.join(head, '%010d.png' % (number + 1))
        return os.path.isfile(new_filename) and os.path.isfile(new_filename1)

    if verify:
        if split == 'train':
            idx = 0
            while idx < len(paths_d):
                filename = paths_d[idx]
                if not verify_nearby(filename):
                    del paths_d[idx]
                    del paths_rgb[idx]
                    del paths_gt[idx]
                else:
                    idx += 1

    print('paths_d: ', len(paths_d))
    print('paths_rgb: ', len(paths_rgb))
    print('paths_gt: ', len(paths_gt))

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0 and use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and use_rgb:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and use_g:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise (RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths


class CompletionDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 val_split='select',
                 opt=None,
                 inf=False):
        super(CompletionDataset, self).__init__()

        self.data_path = data_path
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.val_split = val_split

        self.opt = opt

        self.not_full_res = self.opt.completion_not_full_res

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = not opt.completion_test
        if is_train:
            self.split = 'train'
        else:
            self.split = 'val'
        if opt.completion_test:
            self.split = 'test_completion'
        self.paths = get_paths_and_transform(self.data_path, self.split, self.val_split)

        if inf:
            paths_rgb = [
                'kitti_data/completion/data_rgb/train/2011_09_26_drive_0106_sync/image_02/data/0000000015.png',
                'kitti_data/completion/data_rgb/train/2011_09_26_drive_0096_sync/image_02/data/0000000228.png',
            ]
            paths_d = [
                'kitti_data/completion/data_depth_velodyne/train/2011_09_26_drive_0106_sync/proj_depth/velodyne_raw/image_02/0000000015.png',
                'kitti_data/completion/data_depth_velodyne/train/2011_09_26_drive_0096_sync/proj_depth/velodyne_raw/image_02/0000000228.png',
            ]
            paths_gt = [
                'kitti_data/completion/data_depth_annotated/train/2011_09_26_drive_0106_sync/proj_depth/groundtruth/image_02/0000000015.png',
                'kitti_data/completion/data_depth_annotated/train/2011_09_26_drive_0096_sync/proj_depth/groundtruth/image_02/0000000228.png',
            ]
            self.paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}

    def bottom_crop(self, img):
        #print('in: ', img.shape)
        #cv2.imwrite('in.jpg', img)
        h = img.shape[0]
        w = img.shape[1]
        th, tw = 352, 1216
        i = h - th
        j = int(round((w - tw) / 2.))
        if img.ndim == 3:
            img = img[i:i + th, j:j + tw, :]
        elif img.ndim == 2:
            img = img[i:i + th, j:j + tw]
        #print('out: ', img.shape)
        #cv2.imwrite('out.jpg', img)
        return img



    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.paths['rgb'])

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        if self.opt is not None:
            if self.opt.eval_gdc:
                if self.is_train:
                    inputs['date'] = self.paths['rgb'][index].split('/')[-4][:10]
                else:
                    inputs['date'] = self.paths['rgb'][index].split('/')[-1][:10]
            if self.opt.need_path:
                inputs['path'] = self.paths['rgb'][index]

        if self.is_train:
            head, tail = os.path.split(self.paths['rgb'][index])
            frame_index = int(tail[0:tail.find('.')])
            head_d, _ = os.path.split(self.paths['d'][index])
            for i in self.frame_idxs:
                inputs[("color", i, -1)] = self.get_color(os.path.join(head, '%010d.png' % (frame_index + i)),
                                                          do_flip, padding=self.not_full_res)
                if self.opt.completion_need2channel=='true':
                    inputs[("2channel", i, 0)] = self.load_4beam_2channel(os.path.join(head_d,
                                                                          '%010d.png' % (frame_index + i)), do_flip,
                                                                          padding=self.not_full_res,
                                                                          pool=self.not_full_res)
                else:
                    sparse_depth = self.get_depth(os.path.join(head_d, '%010d.png' % (frame_index + i)),
                                                  do_flip, padding=self.not_full_res, pool=self.not_full_res) / 100.0
                    inputs[("2channel", i, 0)] = torch.stack([sparse_depth[0], sparse_depth[0]])
        else:
            inputs[("color", 0, -1)] = self.get_color(self.paths['rgb'][index], do_flip, padding=self.not_full_res)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            inputs["depth_gt"] = self.get_depth(self.paths['gt'][index], do_flip, padding=self.not_full_res, pool=False)


        if self.opt.need_4beam:
            inputs["4beam"] = self.get_depth(self.paths['d'][index], do_flip,
                                             padding=self.not_full_res, pool=self.not_full_res)
            inputs["4beam"] = inputs["4beam"] / 100.0
            if self.opt.eval_gdc:
                inputs['full_res_4beam'] = self.get_depth(self.paths['d'][index], do_flip, pool=False)
            if self.opt.completion_need2channel=='true':
                inputs["2channel"] = self.load_4beam_2channel(self.paths['d'][index], do_flip,
                                                              padding=self.not_full_res, pool=self.not_full_res)
            else:
                inputs["2channel"] = torch.stack([inputs["4beam"][0], inputs["4beam"][0]])

        return inputs

    def get_color(self, file_path, do_flip, padding):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, file_path, do_flip, padding=True, pool=True):
        raise NotImplementedError

    def get_4beam(self, folder, frame_index, side, do_flip, need_full_res):
        raise NotImplementedError

    def compute_4beam_2channel(self, fourbeam, height=192, width=640, expand=2):
        raise NotImplementedError

    def load_4beam_2channel(self, file_path, do_flip, padding=True, pool=True):
        raise NotImplementedError

    def load_pred_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def load_gdc(self, folder, frame_index, side, do_flip, scale):
        raise NotImplementedError
