import os
import sys
import cv2
import torch
import skimage.transform
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from kitti_utils import generate_depth_map
from multiprocessing import Process, Queue, Pool, cpu_count

regenerate = False
if sys.argv[1] == 'regen':
    regenerate = True
    print("regenerating, will clean previous files")
if sys.argv[2] == 'r200':
    input_folder = 'random200'
    output_folder = 'r200_2cha'
    print("for random 200 points sample")
elif sys.argv[2] == '4beam':
    input_folder = '4beam'
    output_folder = '2channel'
    print("for 4-beams sample")


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


def get_4beam(folder, frame_index, side, do_flip):
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    calib_path = get_detec_calib(folder, frame_index)

    velo_filename = os.path.join(
        folder,
        "{}/{:06d}.bin".format(input_folder, int(frame_index)))

    depth_gt = generate_depth_map(calib_path, velo_filename, side_map[side], shape=[384, 1280])
    depth_gt = F.max_pool2d(torch.tensor(depth_gt).unsqueeze(0),
                            2, ceil_mode=True).squeeze().numpy()

    if do_flip:
        depth_gt = np.fliplr(depth_gt)

    return depth_gt


def get_4beam_2channel(fourbeam, height=192, width=640, expand=2):
    expanded_depth = torch.zeros([height, width], dtype=torch.float32)
    confidence_map = torch.zeros([height, width], dtype=torch.float32)
    accumulate = torch.zeros([height, width], dtype=torch.float32)
    for i in range(76, 190):
        for j in range(2, 638):
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

data_path = 'kitti_data/'


def gen2channel(line):
    folder = data_path + line[:-1].split()[0]
    idx = int(line[:-1].split()[1])
    side = line[:-1].split()[2]
    out_path = folder+'/{}'.format(output_folder)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if not regenerate:
        if os.path.isfile(out_path + '/{}_{}_{}.npy'.format(idx, side, False)):
            if os.path.isfile(out_path + '/{}_{}_{}.npy'.format(idx, side, True)):
                return

    four_beam = get_4beam(folder, idx, side, False)
    flip_four_beam = get_4beam(folder, idx, side, True)

    four_beam = torch.from_numpy(four_beam.astype(np.float32))/100.0
    flip_four_beam = torch.from_numpy(flip_four_beam.astype(np.float32))/100.0

    expanded_depth, confidence_map = get_4beam_2channel(four_beam)
    two_channel = torch.stack([expanded_depth, confidence_map]).numpy()
    expanded_depth, confidence_map = get_4beam_2channel(flip_four_beam)
    flip_two_channel = torch.stack([expanded_depth, confidence_map]).numpy()

    np.save(out_path+'/{}_{}_{}.npy'.format(idx, side, False), two_channel)
    np.save(out_path + '/{}_{}_{}.npy'.format(idx, side, True), flip_two_channel)
    #print(out_path+'/{}_{}'.format(idx, side))

def update(*a):
    pbar.update()

test_file_path = 'splits/detection/test.txt'
test_file = open(test_file_path, 'r')
lines = test_file.readlines()
test_file.close()


print("using {} cpu cores".format(cpu_count()))
pool = Pool(cpu_count())
pbar = tqdm(total=len(lines))

for line in lines:
    #gen2channel(line)
    pool.apply_async(gen2channel, args=(line,), callback=update)

pool.close()
pool.join()
pbar.clear(nolock=False)
pbar.close()
