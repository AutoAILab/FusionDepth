import os
import sys
import cv2
import torch
import skimage.transform
import numpy as np
import PIL.Image as pil
import torch.nn.functional as F
from tqdm.auto import tqdm
from kitti_utils import generate_depth_map
from multiprocessing import Process, Queue, Pool
from datasets.completion_dataset import get_paths_and_transform


output_folder = '2cha'


def bottom_crop(img):
    h = img.shape[0]
    w = img.shape[1]
    th, tw = 352, 1216
    i = h - th
    j = int(round((w - tw) / 2.))
    if img.ndim == 3:
        img = img[i:i + th, j:j + tw, :]
    elif img.ndim == 2:
        img = img[i:i + th, j:j + tw]
    return img


def get_depth(file_path):
    assert os.path.exists(file_path), "file not found: {}".format(file_path)
    img_file = pil.open(file_path)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), file_path)

    depth = depth_png.astype(np.float32) / 256.

    depth = bottom_crop(depth)
    depth = depth.copy()

    depth = torch.tensor(depth)

    return depth


def get_4beam_2channel(fourbeam, height=352, width=1216, expand=2):
    expanded_depth = torch.zeros([height, width], dtype=torch.float32)
    confidence_map = torch.zeros([height, width], dtype=torch.float32)
    accumulate = torch.zeros([height, width], dtype=torch.float32)
    for i in range(110, 350):
        for j in range(2, 1214):
            if fourbeam[i][j] != 0:
                expanded_depth[i][j] = fourbeam[i][j]
                confidence_map[i][j] = 1
                accumulate[i][j] = 1
                for dis in range(1, expand+1):
                    confidence = 1/(dis+1)
                    for horizontal in range(1, dis+1):
                        x = horizontal
                        y = dis - horizontal
                        if accumulate[i+x][j+y] == 0 or confidence_map[i+x][j+y] < confidence:
                            expanded_depth[i+x][j+y] = fourbeam[i][j]
                            confidence_map[i+x][j+y] = confidence
                            accumulate[i+x][j+y] = 1
                        elif confidence_map[i+x][j+y] == confidence:
                            expanded_depth[i + x][j + y] += fourbeam[i][j]
                            accumulate[i + x][j + y] += 1

                        if x != 0:
                            x = -horizontal
                            y = dis - horizontal
                            if accumulate[i + x][j + y] == 0 or confidence_map[i + x][j + y] < confidence:
                                expanded_depth[i + x][j + y] = fourbeam[i][j]
                                confidence_map[i + x][j + y] = confidence
                                accumulate[i + x][j + y] = 1
                            elif confidence_map[i + x][j + y] == confidence:
                                expanded_depth[i + x][j + y] += fourbeam[i][j]
                                accumulate[i + x][j + y] += 1

                        if y != 0:
                            x = horizontal
                            y = horizontal - dis
                            if accumulate[i + x][j + y] == 0 or confidence_map[i + x][j + y] < confidence:
                                expanded_depth[i + x][j + y] = fourbeam[i][j]
                                confidence_map[i + x][j + y] = confidence
                                accumulate[i + x][j + y] = 1
                            elif confidence_map[i + x][j + y] == confidence:
                                expanded_depth[i + x][j + y] += fourbeam[i][j]
                                accumulate[i + x][j + y] += 1

                        if x != 0 and y != 0:
                            x = -horizontal
                            y = horizontal - dis
                            if accumulate[i + x][j + y] == 0 or confidence_map[i + x][j + y] < confidence:
                                expanded_depth[i + x][j + y] = fourbeam[i][j]
                                confidence_map[i + x][j + y] = confidence
                                accumulate[i + x][j + y] = 1
                            elif confidence_map[i + x][j + y] == confidence:
                                expanded_depth[i + x][j + y] += fourbeam[i][j]
                                accumulate[i + x][j + y] += 1
    accumulate[accumulate == 0] = 1
    expanded_depth = torch.div(expanded_depth, accumulate)
    return expanded_depth, confidence_map



def gen2channel(path):
    head, tail = os.path.split(path)
    head, _ = os.path.split(head)
    tail = tail[0:tail.find('.')]
    out_path = head+'/{}'.format(output_folder)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if os.path.isfile(out_path + '/{}.npy'.format(tail)):
        return

    four_beam = get_depth(path) / 100.0

    expanded_depth, confidence_map = get_4beam_2channel(four_beam, height=352, width=1216)
    two_channel = torch.stack([expanded_depth, confidence_map]).numpy()

    # cv2.imwrite('ori.jpg', four_beam.numpy()*255)
    # cv2.imwrite('expand.jpg', expanded_depth.numpy()*255)
    # cv2.imwrite('confi.jpg', confidence_map.numpy()*255)

    np.save(out_path + '/{}.npy'.format(tail), two_channel)

def update(*a):
    pbar.update()


paths = get_paths_and_transform('kitti_data/completion', 'train', 'select', verify=False)['d']
paths += get_paths_and_transform('kitti_data/completion', 'val', 'select')['d']


pool = Pool(20)
pbar = tqdm(total=len(paths))

for path in paths:
    #gen2channel(line)
    pool.apply_async(gen2channel, args=(path,), callback=update)

pool.close()
pool.join()
pbar.clear(nolock=False)
pbar.close()
