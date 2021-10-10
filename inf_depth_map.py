# python inf_depth_map.py --load_weights_folder log/newnewencoder/models/weights_best --need_path
from __future__ import absolute_import, division, print_function

import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from layers import disp_to_depth


class Infer:
    def __init__(self, opts):
        self.opt = opts
        self.opt.batch_size = 1
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find a folder at {}".format(self.opt.load_weights_folder)

        print("-> Loading weights from {}".format(self.opt.load_weights_folder))

        self.models = {}

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, False,
            cat4beam_to_color=self.opt.cat_4beam_to_color,
            cat2channel=self.opt.cat2start)
        encoder_path = os.path.join(self.opt.load_weights_folder, "encoder.pth")
        encoder_dict = torch.load(encoder_path)
        model_dict = self.models["encoder"].state_dict()
        self.models["encoder"].load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        self.models["encoder"].to(self.device)
        self.models["encoder"].eval()
        for param in self.models['encoder'].parameters():
            param.requires_grad = False

        if self.opt.beam_encoder:
            self.models["beam_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, False,
                beam_encoder=True)
            beam_encoder_path = os.path.join(self.opt.load_weights_folder, "beam_encoder.pth")
            self.models["beam_encoder"].load_state_dict(torch.load(beam_encoder_path))
            self.models["beam_encoder"].to(self.device)
            self.models["beam_encoder"].eval()
            for param in self.models['beam_encoder'].parameters():
                param.requires_grad = False

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales, cat2end=self.opt.cat2end)
        depth_path = os.path.join(self.opt.load_weights_folder, "depth.pth")
        self.models["depth"].load_state_dict(torch.load(depth_path))
        self.models["depth"].to(self.device)
        self.models["depth"].eval()
        for param in self.models['depth'].parameters():
            param.requires_grad = False

        print("model named:\n  ", self.opt.model_name)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, opt=self.opt)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)

        test_fpath = os.path.join(os.path.dirname(__file__), "splits", "eigen", "{}_files.txt")
        test_filenames = readlines(test_fpath.format("test"))
        test_dataset = self.dataset(
            self.opt.data_path, test_filenames, self.opt.height, self.opt.width,
            [0], 4, is_train=False, img_ext=img_ext, opt=self.opt)
        self.test_loader = DataLoader(
            test_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items\n".format(
            len(train_dataset)))

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def run_epoch(self, split):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_eval()
        if split == 'train':
            loader = self.train_loader
        elif split == 'test':
            loader = self.test_loader

        for batch_idx, inputs in enumerate(loader):

            outputs = self.process_batch(inputs)
            path = inputs['path'][0]
            data_path = 'kitti_data/'
            folder = data_path + path.split()[0]
            idx = int(path.split()[1])
            side = path.split()[2]
            if not self.opt.random_sample > 0:
                out_path = folder + '/inf_depth_{}beam/'.format(self.opt.nbeams)
            else:
                out_path = folder + '/inf_depth_r{}/'.format(self.opt.random_sample)
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            print(path)
            print(batch_idx)
            np.save(out_path + '/{}_{}.npy'.format(idx, side), outputs[("disp", 0)].cpu())
            #np.save(out_path + '/T_1_{}_{}.npy'.format(idx, side), outputs[("cam_T_cam", 0, 1)].cpu())
            #np.save(out_path + '/T_-1_{}_{}.npy'.format(idx, side), outputs[("cam_T_cam", 0, -1)].cpu())



    def process_batch(self, inputs, val=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if key != 'path':
                inputs[key] = ipt.to(self.device)

        features = self.models["encoder"](inputs["color_aug", 0, 0])

        if self.opt.beam_encoder:
            beam_features = self.models["beam_encoder"](inputs["2channel"])
            outputs = self.models["depth"](features, beam_features=beam_features)

        return outputs


from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()

if __name__ == "__main__":
    infer = Infer(opts)
    infer.run_epoch('train')
    infer.run_epoch('test')