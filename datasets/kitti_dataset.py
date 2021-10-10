from __future__ import absolute_import, division, print_function
import cv2
import os
import torch
import skimage.transform
import numpy as np
import PIL.Image as pil
import torch.nn.functional as F
from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


def get_detec_calib(folder, frame_idx):
    img_path = os.path.join(folder, "image_02/data/{:06d}.png".format(int(frame_idx)))
    imshape = cv2.imread(img_path).shape[:2]
    if imshape == (375, 1242):
        return 'kitti_data/2011_09_26'
    if imshape == (370, 1224):
        return 'kitti_data/2011_09_28'
    if imshape == (374, 1238):
        return 'kitti_data/2011_09_29'
    if imshape == (370, 1226):
        return 'kitti_data/2011_09_30'
    if imshape == (376, 1241):
        return 'kitti_data/2011_10_03'


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """

    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side], shape=[375, 1242])

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_4beam(self, folder, frame_index, side, do_flip, need_full_res=False):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        if not self.opt.random_sample > 0:
            folder_name = '{}beam'.format(self.opt.nbeams)
        else:
            folder_name = 'random{}'.format(self.opt.random_sample)
        velo_filename = os.path.join(
            self.data_path,
            folder,
            "{}/{:010d}.bin".format(folder_name, int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side], shape=[384, 1280])
        depth_gt = F.max_pool2d(torch.tensor(depth_gt).unsqueeze(0),
                                2, ceil_mode=True).squeeze().numpy()

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        if need_full_res:
            depth_gt_full = generate_depth_map(calib_path, velo_filename,
                                               self.side_map[side], shape=[375, 1242])
        else:
            depth_gt_full = depth_gt
        return depth_gt, depth_gt_full

    def load_4beam_2channel(self, folder, frame_index, side, do_flip):
        if not self.opt.random_sample > 0:
            folder_name = '2channel{}beam'.format(self.opt.nbeams)
            if self.opt.nbeams == 4:
                folder_name = '2channel'
        else:
            folder_name = 'r{}_2cha'.format(self.opt.random_sample)
        filename = os.path.join(
            self.data_path,
            folder,
            "{}/{}_{}_{}.npy".format(folder_name, int(frame_index), side, do_flip))

        twochannel = np.load(filename)

        twochannel = torch.from_numpy(twochannel.astype(np.float32))

        return twochannel


    def load_pred_depth(self, folder, frame_index, side, do_flip):
        if not self.opt.random_sample > 0:
            depth_path = 'inf_depth_{}beam'.format(self.opt.nbeams)
        else:
            depth_path = 'inf_depth_r{}'.format(self.opt.random_sample)
        filename = os.path.join(
            self.data_path,
            folder,
            "{}/{}_{}.npy".format(depth_path, int(frame_index), side))

        depth_map = np.load(filename)
        depth_map = torch.from_numpy(depth_map.astype(np.float32))[0][0]
        if do_flip:
            depth_map = torch.fliplr(depth_map)
        return depth_map.unsqueeze(0)

    def load_gdc(self, folder, frame_index, side, do_flip, scale):
        if not self.opt.random_sample > 0:
            gdc_path = 'inf_gdc_{}beam'.format(self.opt.nbeams)
        else:
            gdc_path = 'inf_gdc_r{}'.format(self.opt.random_sample)
        if scale == 0:
            filename = os.path.join(self.data_path, folder,
                                    "{}/{}_{}.npy".format(gdc_path, int(frame_index), side))
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


class KITTIDetecDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """

    def __init__(self, *args, **kwargs):
        super(KITTIDetecDataset, self).__init__(*args, **kwargs)

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:06d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = get_detec_calib('kitti_data/'+folder, frame_index)

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:06d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side], shape=[375, 1242])

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_4beam(self, folder, frame_index, side, do_flip, need_full_res=False):
        calib_path = get_detec_calib('kitti_data/'+folder, frame_index)

        folder_name = "4beam"
        if self.opt.random_sample != -1:
            folder_name = "random{}".format(self.opt.random_sample)
        velo_filename = os.path.join(
            self.data_path,
            folder,
            "{}/{:06d}.bin".format(folder_name, int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side], shape=[384, 1280])
        depth_gt = F.max_pool2d(torch.tensor(depth_gt).unsqueeze(0),
                                2, ceil_mode=True).squeeze().numpy()

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        if need_full_res:
            depth_gt_full = generate_depth_map(calib_path, velo_filename,
                                               self.side_map[side], shape=[375, 1242])
        else:
            depth_gt_full = depth_gt
        return depth_gt, depth_gt_full

    def load_4beam_2channel(self, folder, frame_index, side, do_flip):
        folder_name = '2channel'
        if self.opt.random_sample != -1:
            folder_name = "r{}_2cha".format(self.opt.random_sample)
        filename = os.path.join(
            self.data_path,
            folder,
            "{}/{}_{}_{}.npy".format(folder_name, int(frame_index), side, do_flip))

        twochannel = np.load(filename)

        twochannel = torch.from_numpy(twochannel.astype(np.float32))

        return twochannel


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


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """

    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """

    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
