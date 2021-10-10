#python inf_gdc.py
import os
import sys
import cv2
import torch
import argparse
import skimage.transform
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from kitti_utils import generate_depth_map
from multiprocessing import Process, Queue, Pool, cpu_count
from kitti_util_from_pse import Calibration
from gdc_old import GDC
from layers import disp_to_depth




data_path = 'kitti_data/'
train_file_path = 'splits/eigen_zhou/train_files.txt'
train_file = open(train_file_path, 'r')
val_file_path = 'splits/eigen_zhou/val_files.txt'
val_file = open(val_file_path, 'r')
test_file_path = 'splits/eigen/test_files.txt'
test_file = open(test_file_path, 'r')
#lines = train_file.readlines() + val_file.readlines()
lines = train_file.readlines() + test_file.readlines()  # both

def get_gt(folder, idx):
    if not args.random_sample > 0:
        gt_path = data_path + folder + '/{}beam'.format(args.nbeams)
    else:
        gt_path = data_path + folder + '/random{}'.format(args.random_sample)
    calib_dir = os.path.join(data_path, folder.split("/")[0])
    velo_filename = os.path.join(gt_path, "{:010d}.bin".format(idx))
    gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
    return gt_depth


def inf_gdc(line, args):
    folder = line[:-1].split()[0]
    idx = int(line[:-1].split()[1])
    side = line[:-1].split()[2]
    if not args.random_sample > 0:
        out_path = data_path + folder + '/inf_gdc_{}beam'.format(args.nbeams)
    else:
        out_path = data_path + folder + '/inf_gdc_r{}'.format(args.random_sample)

    try:
        date = folder.split('/')[0]
        calib_path = 'kitti_data/{}/calib_cam_to_cam.txt'.format(date)
        calib = Calibration(calib_path)
        gtd = get_gt(folder, idx)
        if not args.random_sample > 0:
            depth_path = data_path + folder + '/inf_depth_{}beam'.format(args.nbeams)
        else:
            depth_path = data_path + folder + '/inf_depth_r{}'.format(args.random_sample)
        pred_disp = np.load(depth_path + '/{}_{}.npy'.format(idx, side))[0][0]
        pred_disp, _ = disp_to_depth(pred_disp, 0.1, 100.0)
        gt_height, gt_width = gtd.shape[:2]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = np.logical_and(gtd > 1e-3, gtd < 80)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        ratio = np.median(gtd[mask]) / np.median(pred_depth[mask])
        pred_depth *= ratio

        gtd[gtd == 0] = -1
        if args.random_sample == -1:
            consider_range = (-0.1, 4.0)
        else:
            consider_range = (-1.5, 9)

        corrected = GDC(pred_depth, gtd, calib, W_tol=3e-5, recon_tol=5e-4,
                        k=10, method='cg', verbose=False, consider_range=consider_range)
        pred_depth = corrected
    except:
        print("GDC failed")

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    np.save(out_path+'/{}_{}.npy'.format(idx, side), pred_depth)

def update(*a):
    pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate sparse pseudo-LiDAR points")
    parser.add_argument('--random_sample', type=int, default=-1)
    parser.add_argument('--nbeams', default=4, type=int)
    args = parser.parse_args()

    pool = Pool(cpu_count())
    pbar = tqdm(total=len(lines))
    for line in lines:
        pool.apply_async(inf_gdc, args=(line, args), callback=update)

    pool.close()
    pool.join()
    pbar.clear(nolock=False)
    pbar.close()
