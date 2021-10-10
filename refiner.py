
from __future__ import absolute_import, division, print_function

import numpy as np
import time
import math
from layers import disp_to_depth
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json
import wandb

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks


class Refiner:
    def __init__(self, options):
        self.opt = options
        print('refinement for {}beams LiDAR'.format(self.opt.nbeams))
        self.opt.clone_gdc = True
        self.opt.refine_2d = True

        self.opt.num_epochs = (8 * 17) // self.opt.batch_size

        vram = torch.cuda.get_device_properties(0).total_memory
        vram = vram / 1024 ** 3
        if vram < 15:
            self.accumulate_step = 2
        else:
            self.accumulate_step = 1
        if self.opt.batch_size > 8:
            self.accumulate_step *= 2

        self.learning_rate = self.opt.learning_rate * (self.opt.batch_size / 8)
        self.scheduler_step_size = int(self.opt.scheduler_step_size * (8 / self.opt.batch_size))
        self.batch_size = int(self.opt.batch_size / self.accumulate_step)

        wandb.init("monodepth2")
        if self.opt.refine_2d:
            self.eval_scales = self.opt.scales
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.opt.refine_load_weights_folder = os.path.expanduser(self.opt.refine_load_weights_folder)

        assert os.path.isdir(self.opt.refine_load_weights_folder), \
            "Cannot find a folder at {}".format(self.opt.refine_load_weights_folder)

        print("-> Loading weights from {}".format(self.opt.refine_load_weights_folder))

        self.models = {}
        self.parameters_to_train = []

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
        encoder_path = os.path.join(self.opt.refine_load_weights_folder, "encoder.pth")
        encoder_dict = torch.load(encoder_path)
        model_dict = self.models["encoder"].state_dict()
        self.models["encoder"].load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        self.models["encoder"].to(self.device)
        self.models["encoder"].eval()
        if self.opt.train_entire_net:
            self.parameters_to_train += list(self.models["encoder"].parameters())

        if self.opt.beam_encoder:
            self.models["beam_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, False,
                beam_encoder=True)
            beam_encoder_path = os.path.join(self.opt.refine_load_weights_folder, "beam_encoder.pth")
            self.models["beam_encoder"].load_state_dict(torch.load(beam_encoder_path))
            self.models["beam_encoder"].to(self.device)
            self.models["beam_encoder"].eval()
            if self.opt.train_entire_net:
                self.parameters_to_train += list(self.models["beam_encoder"].parameters())

            self.models["beam_encoder_pose"] = networks.ResnetEncoder(
                self.opt.num_layers, False, num_input_images=self.num_pose_frames,
                beam_encoder=True)
            beam_encoder_pose_path = os.path.join(self.opt.refine_load_weights_folder, "beam_encoder_pose.pth")
            self.models["beam_encoder_pose"].load_state_dict(torch.load(beam_encoder_pose_path))
            self.models["beam_encoder_pose"].to(self.device)
            self.models["beam_encoder_pose"].eval()
            if self.opt.train_entire_net:
                self.parameters_to_train += list(self.models["beam_encoder_pose"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales, cat2end=self.opt.cat2end)
        depth_path = os.path.join(self.opt.refine_load_weights_folder, "depth.pth")
        self.models["depth"].load_state_dict(torch.load(depth_path))
        self.models["depth"].to(self.device)
        self.models["depth"].eval()
        if self.opt.train_entire_net:
            self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers, False,
                    num_input_images=self.num_pose_frames)
                pose_encoder_path = os.path.join(self.opt.refine_load_weights_folder, "pose_encoder.pth")
                self.models["pose_encoder"].load_state_dict(torch.load(pose_encoder_path))
                self.models["pose_encoder"].to(self.device)
                self.models["pose_encoder"].eval()
                if self.opt.train_entire_net:
                    self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            pose_path = os.path.join(self.opt.refine_load_weights_folder, "pose.pth")
            self.models["pose"].load_state_dict(torch.load(pose_path))
            self.models["pose"].to(self.device)
            self.models["pose"].eval()
            if self.opt.train_entire_net:
                self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.refine_2d:
            print('refine!')
            self.models['refine2d_decoder'] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
                                                                    self.opt.scales, road=True,
                                                                    catxy=(self.opt.catxy == 'true'),
                                                                    deep=(self.opt.refine2d_deep == 'true'))
            refine2d_decoder_path = os.path.join(self.opt.refine_load_weights_folder, "refine2d_decoder.pth")
            if os.path.exists(refine2d_decoder_path):
                self.models["refine2d_decoder"].load_state_dict(torch.load(refine2d_decoder_path))
                print("loading refine2d_decoder model")
            self.models["refine2d_decoder"].to(self.device)
            self.models["refine2d_decoder"].train()
            self.parameters_to_train += list(self.models["refine2d_decoder"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.scheduler_step_size, 0.1)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, opt=self.opt)
        self.train_loader = DataLoader(
            train_dataset, self.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        #val_dataset = self.dataset(
        #    self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
        #    self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, opt=self.opt)
        test_fpath = os.path.join(os.path.dirname(__file__), "splits", "eigen", "{}_files.txt")
        test_filenames = readlines(test_fpath.format("test"))
        val_dataset = self.dataset(
                      self.opt.data_path, test_filenames, self.opt.height, self.opt.width,
                      [0], 4, is_train=False, img_ext=img_ext, opt=self.opt)
        self.val_loader = DataLoader(
            val_dataset, self.opt.eval_batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        gt_path = os.path.join('splits', 'eigen', "gt_depths.npz")
        self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.catxy = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            if self.opt.catxy == 'true':
                self.catxy["False", scale] = Cat_xy(self.batch_size, h, w)
                self.catxy["False", scale].to(self.device)

                self.catxy["True", scale] = Cat_xy(self.opt.eval_batch_size, h, w)
                self.catxy["True", scale].to(self.device)

            self.project_3d[scale] = Project3D(self.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        self.best = 10.0

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()
        wandb.config.update(self.opt)

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                print('validating')
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
        self.model_lr_scheduler.step()

    def process_batch(self, inputs, val=False):
        """Pass a minibatch through the network and generate images and losses
        """
        #start_time = time.time()
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if not self.opt.train_entire_net:
            with torch.no_grad():
                features = self.models["encoder"](inputs["color_aug", 0, 0])
                beam_features = self.models["beam_encoder"](inputs["2channel"])
                if self.opt.refine_depthnet_with_beam == 'true':
                    outputs = self.models["depth"](features, beam_features=beam_features)
                else:
                    outputs = self.models["depth"](features)


        beam = inputs['4beam']
        two_cha = inputs['2channel']
        disp_0 = outputs[("disp", 0)]
        for scale in self.eval_scales:
            if not self.opt.refine_a0 == 'true':
                disp = outputs[("disp", scale)]
            else:
                disp = disp_0
                disp_0 = F.max_pool2d(disp_0, 2, ceil_mode=True)
            disp640 = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp640, self.opt.min_depth, self.opt.max_depth)

            mask = beam > 0
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :, 78:190, 23:617] = 1  # 375 1242
            mask = mask * crop_mask
            ratio = torch.median(beam[mask] * 100.0) / torch.median(depth[mask]).detach()
            depth *= ratio

            scaled_disp = (F.interpolate(1 / depth, disp.shape[2:],
                                         mode="bilinear", align_corners=False) - 0.01) / 9.9
            if scale != 0:
                two_cha = F.max_pool2d(two_cha, 2, ceil_mode=True)

            if self.opt.catxy == 'true':
                for i in range(scale):
                    depth = F.max_pool2d(depth, 2, ceil_mode=True)
                xyz = self.catxy['{}'.format(val), scale](depth, inputs[("inv_K", scale)])
                outputs[("disp", scale)] = torch.cat([scaled_disp, xyz, two_cha], 1)
            else:
                outputs[("disp", scale)] = torch.cat([scaled_disp, two_cha], 1)


        #inf_time = time.time()
        #print("inference time: ", inf_time - start_time)
        if self.use_pose_net and not val:
            outputs.update(self.predict_poses(inputs, features, self.opt.frame_ids))

        losses = {'loss': 0.0}

        for iter in range(self.opt.refine_iter):
            if self.opt.refine_2d:
                offset_output = self.models["refine2d_decoder"](features, beam_features=beam_features,
                                                                depth_maps=outputs, tanh=self.opt.refine_offset)

                for i in self.opt.scales:
                    outputs[("disp", i)] = offset_output[("disp", i)]

            if val:
                self.generate_images_pred(inputs, outputs, [0])
            else:
                self.generate_images_pred(inputs, outputs, self.opt.frame_ids)

            if self.opt.refine_iter == 1:
                self.opt.refine_iter_gama = 1.0

            if not val:
                losses = self.compute_losses(inputs, outputs, losses,
                                             gama=self.opt.refine_iter_gama**(self.opt.refine_iter - iter),
                                             frame_ids=self.opt.frame_ids)

            #print("loss time: ", time.time() - me_time)
            #print("total time: ", time.time() - start_time)
            #print("")


        return outputs, losses

    def predict_poses(self, inputs, features, frame_ids):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in frame_ids}
                beam_pose_feats = {f_i: inputs["2channel", f_i, 0] for f_i in frame_ids}

            for f_i in frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                        if self.opt.beam_encoder:
                            beam_pose_inputs = [beam_pose_feats[f_i], beam_pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]
                        if self.opt.beam_encoder:
                            beam_pose_inputs = [beam_pose_feats[0], beam_pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        if self.opt.beam_encoder:
                            beam_pose_inputs = [self.models["beam_encoder_pose"](torch.cat(beam_pose_inputs, 1))]

                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    if self.opt.beam_encoder:
                        axisangle, translation = self.models["pose"](pose_inputs, beam_inputs=beam_pose_inputs)
                    else:
                        axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        losses = {}
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = 0.0
        total_batches = 0.0

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                total_batches += 1.0
                outputs, _ = self.process_batch(inputs, val=True)

                if "depth_gt" in inputs:
                    gt_depth = self.gt_depths[batch_idx]
                    inputs['depth_gt'] = torch.tensor(gt_depth).unsqueeze(0).unsqueeze(0).cuda()
                    self.compute_depth_losses(inputs, outputs, losses, accumulate=True)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] /= total_batches
        self.log("val", inputs, outputs, losses, val=True)
        print(losses["de/abs_rel"])
        if losses["de/abs_rel"] < self.best:
            self.best = losses["de/abs_rel"]
            print("saving best result: ", self.best)
            self.save_model("best")
            for l, v in losses.items():
                wandb.log({"{}".format(l): v}, step=self.step)
                print(l, v)
            absrel = math.floor(losses["de/abs_rel"] * 10000)
            print('floor', absrel)
            if absrel < 800:
                self.save_model('refine{}'.format(absrel))
                wandb.alert(title="Refiner", text="Refiner achieved abs rel of {}".format(losses["de/abs_rel"]))
        del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, frame_ids):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.eval_scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def siloss(self, pred, target, scale):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        valid_mask = ((target > 1e-3) * (pred < 80) * (pred > 1e-3) *
                      (abs(pred - target) < self.opt.gdc_loss_threshold)).detach()
        d = torch.log(pred[valid_mask]) - torch.log(target[valid_mask])
        return torch.sqrt((d ** 2).mean() - self.opt.si_var * (d.mean() ** 2)) * 10.0

    def compute_l2_loss(self, pred, target, scale):
        """Compute loss between pred depth and 4-beam lidar gt"""
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        mask = (target > 1e-3) * (pred < 80) * (pred > 1e-3)
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, 78:190, 23:617] = 1  # 375 1242
        mask = (mask * crop_mask).detach()
        #try:
            #print(scale, pred.mean())
            #ratio = torch.median(target.clone().detach()[mask]) / torch.median(pred.clone().detach()[mask])
            #ratio = ratio.clone().detach()
            #pred *= ratio
            #print(ratio)
        #except:
        #    print("failed calc mask")


        valid_mask = ((target > 1e-3) * (pred < 80) * (pred > 1e-3) *
                      (abs(pred - target) < self.opt.gdc_loss_threshold)).detach()
        diff = target - pred
        diff = diff[valid_mask]
        loss = (diff ** 2).mean()
        #loss = diff.mean()
        return loss


    def compute_losses(self, inputs, outputs, losses, gama=1.0, frame_ids=[0]):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        total_loss = 0

        for scale in self.eval_scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)].clone()
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/gama{}_scale{}".format(gama, scale)] = loss

            if (not self.opt.gdc_loss_only_on_scale_0) or scale == 0:
                gdc_out = inputs['inf_gdc'].squeeze()
                disp = F.interpolate(disp, [192, 640], mode="bilinear", align_corners=False).squeeze()
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                #depth *= 26.0
                gdc_loss = self.siloss(depth, gdc_out, scale) * self.opt.gdc_loss_weight
                if self.opt.gdc_loss_only_on_scale_0:
                    gdc_loss *= 4.0
                total_loss += gdc_loss
                # print("gdc loss : ", gdc_loss)
                losses["loss/gdc_scale{}".format(scale)] = gdc_loss

        total_loss /= self.num_scales

        losses["loss"] += total_loss * gama
        return losses

    def compute_depth_losses(self, inputs, outputs, losses, accumulate=False):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        gt_height, gt_width = inputs['depth_gt'].shape[2:]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [gt_height, gt_width], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            if accumulate:
                losses[metric] += np.array(depth_errors[i].cpu())
            else:
                losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses, val=False):
        """Write an event to the tensorboard events file
        """
        if not val:
            frame_ids = self.opt.frame_ids
        else:
            frame_ids = [0]

        for l, v in losses.items():
            wandb.log({"{}_{}".format(mode, l): v}, step=self.step)
            #writer.add_scalar("{}_{}".format(mode, l), v, self.step)

        for j in range(min(0, inputs['4beam'].shape[0])):  # write a maxmimum of four images
            for s in self.eval_scales:
                for frame_id in frame_ids:
                    wandb.log({"{}_color_{}_{}/{}".format(mode, frame_id, s, j):
                               [wandb.Image(inputs[("color", frame_id, s)][j].data)]}, step=self.step)
                    if s == 0 and frame_id != 0:
                        wandb.log({"{}_color_pred_{}_{}/{}".format(mode, frame_id, s, j):
                                  [wandb.Image(outputs[("color", frame_id, s)][j].data)]}, step=self.step)

                wandb.log({"{}_disp_{}/{}".format(mode, s, j):
                               [wandb.Image(normalize_image(outputs[("disp", s)][j]))]}, step=self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(frame_ids[1:]):
                        wandb.log({"{}_predictive_mask_{}_{}/{}".format(mode, frame_id, s, j):
                                  [wandb.Image(outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...])]},
                                  step=self.step)

                elif not self.opt.disable_automasking and not val:
                    wandb.log({"{}_automask_{}/{}".format(mode, s, j):
                              [wandb.Image(outputs["identity_selection/{}".format(s)][j][None, ...])]},
                              step=self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, folder=None):
        """Save model weights to disk
        """
        if folder == None:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        else:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(folder))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)


from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    refiner = Refiner(opts)
    refiner.train()