import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import cv2
import torch
import skimage.transform
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from kitti_utils import generate_depth_map, ptc2depth
from multiprocessing import Process, Queue, Pool, cpu_count

regenerate = False
test_only = False
demo = False
minus1only = False

if len(sys.argv) > 3:
    if sys.argv[3] == 'test_only':
        test_only = True
    elif sys.argv[3] == 'demo':
        demo = True
    elif sys.argv[3] == '-1only':
        minus1only = True

if sys.argv[1] == 'regen':
    regenerate = True
    print("regenerating, will clean previous files")
if sys.argv[2] == 'r200':
    input_folder = 'random200'
    output_folder = 'r200_2cha'
    print("for random 200 points sample")
if sys.argv[2] == 'r100':
    input_folder = 'random100'
    output_folder = 'r100_2cha'
    print("for random 200 points sample")
elif sys.argv[2] == '4beam':
    input_folder = '4beam'
    output_folder = '2channel'
    print("for 4-beams sample")
elif sys.argv[2] == '1beam' or sys.argv[2] == '2beam' or sys.argv[2] == '3beam' or sys.argv[2] == '8beam' \
     or sys.argv[2] == '16beam':
    input_folder = sys.argv[2]
    output_folder = '2channel{}'.format(sys.argv[2])
    print("for {} sample".format(sys.argv[2]))
elif sys.argv[2] == '4beamT':
    input_folder = 'inf_translidar_4beam'
    if minus1only:
        output_folder = '2cha_4beamT-1'
        print("for T-1 frame 4-beam sample")
    else:
        output_folder = '2cha_4beamT'
        print("for 2 frame temporal fusion 4-beam sample")
elif sys.argv[2] == '4beamT2':
    input_folder = 'inf_translidar2_4beam'
    if minus1only:
        output_folder = '2cha_4beamT-2'
        print("for T-1 frame 4-beam sample")
    else:
        output_folder = '2cha_4beamT2'
        print("for 2 frame temporal fusion 4-beam sample")

def get_4beam(folder, frame_index, side, do_flip):
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    calib_path = os.path.join(folder.split("/")[0]+'/'+folder.split("/")[1])

    if input_folder == 'inf_translidar_4beam' or input_folder == 'inf_translidar2_4beam':
        if minus1only:
            ptc_filename = folder + "/{}/{}_{}_1t.npy".format(input_folder, int(frame_index), side)
        else:
            ptc_filename = folder + "/{}/{}_{}.npy".format(input_folder, int(frame_index), side)
        ptc = np.load(ptc_filename)
        depth_gt = ptc2depth(calib_path, ptc, cam=side_map[side], shape=[384, 1280])
    else:
        velo_filename = os.path.join(
            folder,
            "{}/{:010d}.bin".format(input_folder, int(frame_index)))

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

    '''
    mask = (four_beam > 0)
    crop_mask = torch.zeros_like(mask)
    crop_mask[78:190, 23:617] = 1  # 375 1242
    mask = mask * crop_mask
    print(two_channel[0, :, :][mask] / four_beam[mask])
    print(mask)
    for i in range(192):
        for j in range(640):
            if two_channel[0,i,j] != four_beam[i,j] and mask[i,j]:
                print(i, j, two_channel[0,i,j], four_beam[i,j])
            if two_channel[1,i,j] != 1 and mask[i,j]:
                print("not one ",i, j, two_channel[1,i,j])

    cv2.imwrite('4beam.jpg', flip_four_beam.numpy()*255)
    cv2.imwrite('expanded.jpg', expanded_depth.numpy()*255)
    cv2.imwrite('confidence.jpg', confidence_map.numpy()*255)

    yy, xx = torch.meshgrid([torch.arange(0, 192, dtype=torch.int32),
                             torch.arange(0, 640, dtype=torch.int32)])
    indices = expanded_depth != 0
    depth_feat = torch.stack((yy[indices], xx[indices], expanded_depth[indices]*255), -1).view([-1, 3])

    indices = flip_four_beam != 0
    beam_feat = torch.stack([yy[indices], xx[indices], flip_four_beam[indices]*255], -1).view([-1, 3])
    import open3d
    pcd_depth = open3d.geometry.PointCloud()
    pcd_depth.points = open3d.utility.Vector3dVector(depth_feat)
    pcd_beam = open3d.geometry.PointCloud()
    pcd_beam.points = open3d.utility.Vector3dVector(beam_feat)
    pcd_beam.paint_uniform_color([0, 0, 0])
    open3d.visualization.draw_geometries([pcd_depth, pcd_beam])
    import pdb
    pdb.set_trace()
    '''

    np.save(out_path+'/{}_{}_{}.npy'.format(idx, side, False), two_channel)
    np.save(out_path + '/{}_{}_{}.npy'.format(idx, side, True), flip_two_channel)
    #print(out_path+'/{}_{}'.format(idx, side))

def update(*a):
    pbar.update()

test_file_path = 'splits/eigen/added_minus1.txt'
test_file = open(test_file_path, 'r')
lines = test_file.readlines()
test_file.close()
if not test_only:
    #train_file_path = 'splits/eigen_zhou/train_files.txt'
    #train_file = open(train_file_path, 'r')
    #val_file_path = 'splits/eigen_zhou/val_files.txt'
    #val_file = open(val_file_path, 'r')
    #lines += train_file.readlines() + val_file.readlines()
    train_file_path = 'splits/eigen_full/train_files.txt'
    train_file = open(train_file_path, 'r')
    val_file_path = 'splits/eigen_full/val_files.txt'
    val_file = open(val_file_path, 'r')
    lines += train_file.readlines() + val_file.readlines()
    lines = list(set(lines))
    train_file.close()
    val_file.close()
if demo:
    demo_file_path = 'splits/demo/demo.txt'
    demo_file = open(demo_file_path, 'r')
    lines = demo_file.readlines()
    demo_file.close()

if __name__ == '__main__':
    print("using {} cpu cores".format(cpu_count()))
    pool = Pool(cpu_count())
    pbar = tqdm(total=len(lines))

    for line in lines:
        #gen2channel(line)
        #pbar.update()
        pool.apply_async(gen2channel, args=(line,), callback=update)

    pool.close()
    pool.join()
    pbar.clear(nolock=False)
    pbar.close()
