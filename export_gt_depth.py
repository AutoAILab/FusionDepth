from __future__ import absolute_import, division, print_function

import os
import cv2
import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map


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


def export_gt_depths_detec():

    parser = argparse.ArgumentParser(description='export_gt_depth')
    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        default='./kitti_data')
    parser.add_argument('--txt_filename',
                        type=str,
                        help='filename to the split txt',
                        default='test_files')
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark", "4beam", "8beam", "16beam",
                                 "3beam", "2beam", "1beam", "r100", "r200", "detec", "detec4beam", "demo", "demo_gt"])
    opt = parser.parse_args()

    if opt.split == '4beam' or opt.split == '1beam' or opt.split == '2beam' or opt.split == '3beam'\
       or opt.split == '16beam' or opt.split == '8beam' or opt.split == 'r200' or opt.split == 'r100':
        split_folder = os.path.join(os.path.dirname(__file__), "splits", "eigen")
    elif opt.split == 'detec' or opt.split == 'detec4beam':
        split_folder = os.path.join(os.path.dirname(__file__), "splits", "detection")
    elif opt.split == 'demo' or opt.split == 'demo_gt':
        split_folder = os.path.join(os.path.dirname(__file__), "splits", "demo")
    else:
        split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "{}.txt".format(opt.txt_filename)))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    for line in lines:

        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        if opt.split == "eigen":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        if opt.split == "detec":
            calib_dir = get_detec_calib(os.path.join(opt.data_path, folder), frame_id)
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:06d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        if opt.split == "demo_gt":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        if opt.split == "detec4beam":
            calib_dir = get_detec_calib(os.path.join(opt.data_path, folder), frame_id)
            velo_filename = os.path.join(opt.data_path, folder,
                                         "4beam", "{:06d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        if opt.split == '4beam' or opt.split == '1beam' or opt.split == '2beam' or opt.split == '3beam' \
           or opt.split == '8beam' or opt.split == '16beam':
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "{}".format(opt.split), "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        if opt.split == "r100":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "random100", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        if opt.split == "r200":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "random200", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        if opt.split == "demo":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "4beam", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256

        gt_depths.append(gt_depth.astype(np.float32))
        # print((gt_depth!=0).sum())

    if opt.split == '4beam' or opt.split == '1beam' or opt.split == '2beam' or opt.split == '3beam' \
       or opt.split == '8beam' or opt.split == '16beam':
        output_path = os.path.join(split_folder, "{}.npz".format(opt.split))
    elif opt.split == 'detec4beam':
        output_path = os.path.join(split_folder, "4beam.npz")
    elif opt.split == 'r200':
        output_path = os.path.join(split_folder, "r200.npz")
    elif opt.split == 'r100':
        output_path = os.path.join(split_folder, "r100.npz")
    elif opt.split == 'demo':
        output_path = os.path.join(split_folder, "4beam.npz")
    else:
        output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_detec()
