from __future__ import absolute_import, division, print_function

import cv2
import os
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


class MonoDataset(data.Dataset):
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
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 opt=None):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.opt = opt

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

        self.load_depth = self.check_depth()

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
        return len(self.filenames)

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

        line = self.filenames[index].split()
        folder = line[0]

        if self.opt is not None:
            inputs['date'] = folder.split('/')[0]
            if self.opt.need_path:
                inputs['path'] = self.filenames[index]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
                if self.opt.need_2_channel:
                    inputs[("2channel", i, 0)] = self.load_4beam_2channel(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if self.opt.need_4beam:
            four_beam, four_beam_full = self.get_4beam(folder, frame_index, side, do_flip,
                                                       need_full_res=self.opt.need_full_res_4beam)
            inputs["4beam"] = np.expand_dims(four_beam, 0)
            inputs["4beam"] = torch.from_numpy(inputs["4beam"].astype(np.float32))
            inputs["4beam"] = inputs["4beam"] / 100.0
            if self.opt.need_full_res_4beam:
                inputs["4beam_full"] = np.expand_dims(four_beam_full, 0)
                inputs["4beam_full"] = torch.from_numpy(inputs["4beam_full"].astype(np.float32))
                inputs["4beam_full"] = inputs["4beam_full"] / 100.0
            if self.opt.need_2_channel:
                #expanded_depth, confidence_map = self.compute_4beam_2channel(inputs["4beam"].clone())
                #inputs["2channel"] = torch.stack([expanded_depth, confidence_map])
                inputs["2channel"] = self.load_4beam_2channel(folder, frame_index, side, do_flip)
                if self.opt.need_full_res_4beam:
                    inputs["2channel_full"] = inputs["2channel"].view(192, 640, 2).numpy()
                    inputs["2channel_full"] = cv2.resize(inputs["2channel_full"], (1242, 375),
                                                         interpolation=cv2.INTER_NEAREST)
                    inputs["2channel_full"] = torch.from_numpy(inputs["2channel_full"]).view(2, 375, 1242)
                #cv2.imwrite('ori_cat.jpg', inputs["4beam"][0].numpy()*255)
                #cv2.imwrite('expanded.jpg', expanded_depth.numpy()*255)
                #cv2.imwrite('confidence.jpg', confidence_map.numpy()*255)

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        if (self.opt.clone_gdc and self.is_train) or self.opt.need_inf_gdc:
            # inputs["inf_depth"] = self.load_pred_depth(folder, frame_index, side, do_flip)
            inputs["inf_gdc"] = self.load_gdc(folder, frame_index, side, do_flip, 0)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_4beam(self, folder, frame_index, side, do_flip, need_full_res):
        raise NotImplementedError

    def load_4beam_2channel(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def load_pred_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def load_gdc(self, folder, frame_index, side, do_flip, scale):
        raise NotImplementedError
