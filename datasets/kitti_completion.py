from __future__ import absolute_import, division, print_function

import os
import torch
import skimage.transform
import numpy as np
import PIL.Image as pil
import torch.nn.functional as F
from kitti_utils import generate_depth_map
from .completion_dataset import CompletionDataset


class KITTICompletion(CompletionDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(KITTICompletion, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def get_color(self, file_path, do_flip, padding):
        color = self.loader(file_path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        if padding:
            color = np.array(color)
            ypad = 384 - color.shape[0]
            xpad = 1280 - color.shape[1]
            xpad1 = xpad // 2
            color = np.pad(color, ((ypad, 0), (xpad1, xpad - xpad1), (0, 0)))
            color = pil.fromarray(color)

        if not self.not_full_res:
            color = np.array(color)
            color = self.bottom_crop(color)
            color = pil.fromarray(color)
        #assert not np.isnan(np.sum(np.array(color))), "color is nan"

        return color

    def get_depth(self, file_path, do_flip, padding=True, pool=True):
        assert os.path.exists(file_path), "file not found: {}".format(file_path)
        img_file = pil.open(file_path)
        depth_png = np.array(img_file, dtype=int)
        img_file.close()
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255, \
            "np.max(depth_png)={}, path={}".format(np.max(depth_png), file_path)

        depth = depth_png.astype(np.float32) / 256.

        #assert not np.isnan(np.sum(np.array(depth))), "depth is nan"

        if do_flip:
            depth = np.fliplr(depth)

        if not self.not_full_res:
            depth = self.bottom_crop(depth)
            depth = depth.copy()

        if padding:
            ypad = 384 - depth.shape[0]
            xpad = 1280 - depth.shape[1]
            xpad1 = xpad // 2
            depth = np.pad(depth, ((ypad, 0), (xpad1, xpad - xpad1)))
        if pool:
            depth = F.max_pool2d(torch.tensor(depth).unsqueeze(0), 2, padding=0, ceil_mode=True)
        else:
            depth = torch.tensor(depth).unsqueeze(0)
        return depth

    def load_4beam_2channel(self, file_path, do_flip, padding=True, pool=True):
        head, tail = os.path.split(file_path)
        head, _ = os.path.split(head)
        tail = tail[0:tail.find('.')]
        npy_path = head + '/2cha/{}.npy'.format(tail)
        assert os.path.exists(npy_path), "file not found: {}".format(npy_path)
        two_cha = np.load(npy_path)

        if do_flip:
            two_cha = np.fliplr(two_cha)

        if padding:
            ypad = 384 - two_cha.shape[0]
            xpad = 1280 - two_cha.shape[1]
            xpad1 = xpad // 2
            two_cha = np.pad(two_cha, ((ypad, 0), (xpad1, xpad - xpad1)))

        two_cha = two_cha.copy()

        if pool:
            two_cha = F.max_pool2d(torch.tensor(two_cha).unsqueeze(0), 2, padding=0, ceil_mode=True)
        else:
            two_cha = torch.tensor(two_cha) # .unsqueeze(0)
        return two_cha

    def load_pred_depth(self, folder, frame_index, side, do_flip):
        filename = os.path.join(
            self.data_path,
            folder,
            "inf_depth/{}_{}.npy".format(int(frame_index), side))

        depth_map = np.load(filename)
        depth_map = torch.from_numpy(depth_map.astype(np.float32))[0][0]
        if do_flip:
            depth_map = torch.fliplr(depth_map)
        return depth_map.unsqueeze(0)

    def load_gdc(self, folder, frame_index, side, do_flip, scale):
        if scale == 0:
            filename = os.path.join(self.data_path, folder,
                                    "inf_gdc/{}_{}.npy".format(int(frame_index), side))
        else:
            filename = os.path.join(self.data_path, folder,
                                    "inf_gdc123/{}_{}_{}.npy".format(int(frame_index), side, scale))

        gdc = np.load(filename)
        gdc = torch.from_numpy(gdc.astype(np.float32))
        gdc = F.interpolate(gdc.unsqueeze(0).unsqueeze(0),
                            [192, 640], mode="bilinear", align_corners=False).squeeze()
        if do_flip:
            gdc = torch.fliplr(gdc)

        return gdc