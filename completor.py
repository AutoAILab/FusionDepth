from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

#import open3d
import json
import wandb

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from layers import disp_to_depth

#from torch.cuda.amp import autocast, GradScaler
from options import MonodepthOptions


class Completor:
    def __init__(self, options):
        self.opt = options
        if not self.opt.completion_not_full_res:
            print('full res')
            self.opt.height = 352
            self.opt.width = 1216
        wandb.init(project="completion")
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

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
            self.opt.completion_num_layers, self.opt.weights_init == "pretrained",
            cat4beam_to_color=self.opt.cat_4beam_to_color,
            cat2channel=self.opt.cat2start)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        if self.opt.beam_encoder:
            self.models["beam_encoder"] = networks.ResnetEncoder(
                self.opt.completion_num_layers, self.opt.weights_init == "pretrained",
                beam_encoder=True)
            self.models["beam_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["beam_encoder"].parameters())

            self.models["beam_encoder_pose"] = networks.ResnetEncoder(
                self.opt.completion_pose_num_layers, self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames, beam_encoder=True)
            self.models["beam_encoder_pose"].to(self.device)
            self.parameters_to_train += list(self.models["beam_encoder_pose"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales, cat2end=self.opt.cat2end)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.completion_pose_num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.completion_scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTICompletion}
        self.dataset = datasets_dict[self.opt.dataset]

        train_dataset = self.dataset(
            self.opt.data_path+'/completion', self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, val_split=self.opt.completion_val_split, opt=self.opt)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = self.dataset(
            self.opt.data_path+'/completion', self.opt.height, self.opt.width,
            [0], 4, is_train=False, val_split=self.opt.completion_val_split, opt=self.opt)
        self.val_loader = DataLoader(
            val_dataset, self.opt.eval_batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        self.val_iter = iter(self.val_loader)

        num_train_samples = len(train_dataset.paths['rgb'])
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.completion_num_epochs

        gt_path = os.path.join('splits', 'eigen', "gt_depths.npz")
        #self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        # self.writers = {}
        # for mode in ["train", "val"]:
        #     self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()
        wandb.config.update(self.opt)
        self.best = 100000.0

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
        #self.scaler = GradScaler()
        torch.autograd.set_detect_anomaly(False)
        for self.epoch in range(self.opt.completion_num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        print(self.model_lr_scheduler.get_last_lr())
        self.set_train()

        data_loading_time = 0
        gpu_time = 0
        before_op_time = time.time()

        for batch_idx, inputs in enumerate(self.train_loader):

            data_loading_time += (time.time() - before_op_time)

            before_op_time = time.time()

            #if self.opt.completion_amp:
            #    with autocast():
            #        outputs, losses = self.process_batch(inputs)
            #        self.model_optimizer.zero_grad()
            #        self.scaler.scale(losses["loss"]).backward()
            #        self.scaler.step(self.model_optimizer)
            #        self.scaler.update()
            #else:
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            # torch.nn.utils.clip_grad_norm_(self.parameters_to_train, 0.01)
            self.model_optimizer.step()

            duration = time.time() - before_op_time
            gpu_time += duration

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(
                    batch_idx, duration, losses["loss"].cpu().data, data_loading_time, gpu_time)
                data_loading_time = 0
                gpu_time = 0

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
            before_op_time = time.time()

        self.model_lr_scheduler.step()

    def process_batch(self, inputs, val=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if key == 'date':
                continue
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder

            if self.opt.cat_4beam_to_color:
                features = self.models["encoder"](torch.cat((inputs["color_aug", 0, 0], inputs["4beam"]), 1))
            elif self.opt.cat2start:
                features = self.models["encoder"](torch.cat((inputs["color_aug", 0, 0], inputs["2channel"]), 1))
            else:
                features = self.models["encoder"](inputs["color_aug", 0, 0])

            if self.opt.cat2end:
                outputs = self.models["depth"](features, two_channel=inputs["2channel"])
            elif self.opt.beam_encoder:
                beam_features = self.models["beam_encoder"](inputs["2channel"])
                outputs = self.models["depth"](features, beam_features=beam_features)
            else:
                outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net and not val:
            outputs.update(self.predict_poses(inputs, features))

        if val:
            self.generate_images_pred(inputs, outputs, [0])
        else:
            self.generate_images_pred(inputs, outputs, self.opt.frame_ids)
        losses = {}
        if not val:
            losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
                beam_pose_feats = {f_i: inputs["2channel", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
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
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
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
                    #gt_depth = self.gt_depths[batch_idx]
                    #inputs['depth_gt'] = torch.tensor(gt_depth).unsqueeze(0).unsqueeze(0).cuda()
                    self.compute_depth_losses(inputs, outputs, losses, accumulate=True)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] /= total_batches
        self.log("val", inputs, outputs, losses, val=True)
        if losses["de/rms"] < self.best:
            self.best = losses["de/rms"]
            print("saving best result: ", self.best)
            for l, v in losses.items():
                wandb.log({"{}".format(l): v}, step=self.step)
                print(l, v)
            # self.save_model("best")
            rms = round(losses["de/rms"])
            if rms < 1200:
                self.save_model('rms{}'.format(rms))
        del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, frame_ids):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
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

                # from the authors of https://arxiv.org/abs/1712.00175
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
                    outputs[("sample", frame_id, scale)].float(),
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

    l1loss = nn.L1Loss()

    def siloss(self, pred, target, scale):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        assert not torch.isnan(pred).any(), "pred nan"
        assert not torch.isnan(target).any(), "target nan"

        #valid_mask = ((target > 1) * (pred < 80) * (pred > 1) *
        #              (abs(pred - target) < self.opt.gdc_loss_threshold)).detach()
        #valid_mask2 = ((target > 1) * (pred < 80) * (pred > 1e-3)).detach()
        #valid_mask1 = (target > 1).detach()
        #print('v', valid_mask.sum())
        #print('v1', valid_mask1.sum())
        #print('v2', valid_mask2.sum())
        #print('pred_median', pred[valid_mask1].median())
        #print(target[valid_mask])
        print('minmax', pred.min(), pred.max(), target.min(), target.max())
        #d = torch.log(pred[valid_mask]) - torch.log(target[valid_mask])
        #d = pred - target
        #print('d', d)
        #si_loss = torch.sqrt((d ** 2).mean() - self.opt.si_var * (d.mean() ** 2)) * 10.0
        si_loss = self.l1loss(pred, target)
        print('si loss', si_loss)
        #print(pred)
        assert not torch.isnan(si_loss).any(), "si_loss nan"
        return si_loss
        #return 1.0

    # def siloss1(self, preds, actual_depth, scale):
    #     n_pixels = actual_depth.shape[2] * actual_depth.shape[3]
    #
    #     invalid_mask = (actual_depth < 1).detach()
    #     # preds[invalid_mask] = 0.00001
    #     # actual_depth[invalid_mask] = 0.00001
    #
    #     preds[preds <= 0] = 0.00001
    #     actual_depth[actual_depth == 0] = 0.00001
    #     d = torch.log(preds) - torch.log(actual_depth)
    #
    #     term_1 = torch.pow(d.view(-1, n_pixels), 2).mean(dim=1).sum()  # pixel wise mean, then batch sum
    #     term_2 = (torch.pow(d.view(-1, n_pixels).sum(dim=1), 2) / (2 * (n_pixels ** 2))).sum()
    #
    #     return term_1 - term_2

    def l2loss(self, pred, target, scale):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        valid_mask = ((target > 1e-3) * (pred < 80) * (pred > 1e-3)).detach()
        diff = target - pred
        diff = diff[valid_mask]
        loss = (diff ** 2).mean()
        return loss


    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
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
            losses["loss/{}".format(scale)] = loss

            if self.opt.completion_siloss_all_scale=="true" or scale == 0:
                # print(disp.shape)
                # print('disp', disp)
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                # print(disp.shape)
                # print('dispi', disp)
                #print(disp)
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                beam_depth = inputs['4beam'] * 100.0

                ################################# Debug
                #if self.opt.debug:
                #    def depth2ptc(depth, calib):
                #        """Convert a depth_map to a pointcloud."""
                #        rows, cols = depth.shape
                #        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
                #        points = np.stack([c, r, depth]).reshape((3, -1)).T
                #        return calib.project_image_to_rect(points)
                #
                #    gt_depth = inputs['depth_gt']
                #    #gt_depth = F.max_pool2d(gt_depth, 2, padding=0, ceil_mode=True)
                #    beam_depthn = inputs['4beam'].cpu().numpy() * 100.0
                #    print(gt_depth.shape)
                #    print(beam_depthn.shape)
                #    from kitti_util_from_pse import Calibration
                #    date = inputs['date'][0]
                #    calib_path = '/media/zfeng/storage/ziyue/datasets/kitti_monod2/kitti_data/{}/calib_cam_to_cam.txt'\
                #        .format(date)
                #    calib = Calibration(calib_path)
                #    beam_points = depth2ptc(beam_depthn[0][0], calib)
                #    beam_pcd = open3d.geometry.PointCloud()
                #    beam_pcd.points = open3d.utility.Vector3dVector(beam_points)
                #    # open3d.io.write_point_cloud('corrected_old.pcd', pcd)
                #    gt_points = depth2ptc(gt_depth[0][0].cpu().numpy(), calib)
                #    gt_pcd = open3d.geometry.PointCloud()
                #    gt_pcd.points = open3d.utility.Vector3dVector(gt_points)
                #
                #    mask = inputs['4beam'] > 0
                #    ratio = torch.median(inputs['4beam'][mask] * 100.0) / torch.median(depth[mask]).detach()
                #    scaled_depth_d = depth * ratio
                #    print('scale ratio:', ratio)
                #
                #    mask = inputs['depth_gt'] > 0
                #    ratio = torch.median(inputs['depth_gt'][mask] * 100.0) / torch.median(depth[mask]).detach()
                #    scaled_depth_gt = depth * ratio
                #
                #    pred_points = depth2ptc(scaled_depth_d[0][0].cpu().detach().numpy(), calib)
                #    pred_pcd = open3d.geometry.PointCloud()
                #    pred_pcd.points = open3d.utility.Vector3dVector(pred_points)
                #
                #    beam_pcd.paint_uniform_color([0, 0, 0])
                #    open3d.visualization.draw_geometries([beam_pcd, pred_pcd])
                #    print('sid', self.siloss(depth, beam_depth, scale))
                #    print('sigt', self.siloss(depth, gt_depth, scale))
                #    print('sisd', self.siloss(scaled_depth_d, beam_depth, scale))
                #    print('sisgt', self.siloss(scaled_depth_gt, gt_depth, scale))
                #
                ###################################

                if not self.opt.completion_siloss_all_scale=="true":
                    self.opt.completion_siloss_weight *= 2.0

                if self.opt.completion_siloss:
                    # print(outputs[('disp', 0)])
                    # gt_depth = inputs['depth_gt']
                    #si_loss = self.l1loss(depth, beam_depth)
                    #si_loss = self.siloss(depth * 26.0, beam_depth, scale) * self.opt.completion_siloss_weight
                    # si_loss = self.siloss(depth * 26.0, gt_depth, scale) * self.opt.completion_siloss_weight
                    depth *= 26.0

                    # valid_mask2 = ((beam_depth > 1) * (depth < 80) * (depth > 1e-3)).detach()
                    # valid_mask1 = (beam_depth > 1).detach()
                    valid_mask = ((beam_depth > 1) * (depth < 80) * (depth > 1) * (abs(depth - beam_depth) < self.opt.gdc_loss_threshold)).detach()

                    #print('v', valid_mask.sum())
                    #print('v1', valid_mask1.sum())
                    #print('v2', valid_mask2.sum())
                    #print('pred_median', depth[valid_mask1].median())
                    #print(beam_depth[valid_mask])
                    #print('minmax', depth.min(), depth.max(), beam_depth.min(), beam_depth.max())
                    d = torch.log(depth[valid_mask]) - torch.log(beam_depth[valid_mask])
                    #print('d', d)
                    si_loss = torch.sqrt((d ** 2).mean() - self.opt.si_var * (d.mean() ** 2)) * 0.1
                    total_loss += si_loss
                    losses["loss/si_loss{}".format(scale)] = si_loss
                elif self.opt.completion_l1loss:
                    depth *= 26.0
                    valid_mask = ((beam_depth > 1) * (depth < 80) * (depth > 1)).detach()
                    l1_loss = self.l1loss(depth[valid_mask], beam_depth[valid_mask]) * 0.001
                    total_loss += l1_loss
                    losses["loss/l1_loss{}".format(scale)] = l1_loss

        total_loss /= self.num_scales

        losses["loss"] = total_loss
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
        mask = depth_gt > 0.1

        # garg/eigen crop
        if self.opt.completion_eigen_crop:
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :, 153:371, 44:1197] = 1
            mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt*1000.0, depth_pred*1000.0)

        for i, metric in enumerate(self.depth_metric_names):
            if accumulate:
                losses[metric] += np.array(depth_errors[i].cpu())
            else:
                losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss, data_time, gpu_time):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {} | CPU/GPU time: {:0.1f}s/{:0.1f}s"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left),
                                  data_time, gpu_time))

    def log(self, mode, inputs, outputs, losses, val=False):
        """Write an event to the tensorboard events file
        """
        # if not val:
        #     frame_ids = self.opt.frame_ids
        # else:
        #     frame_ids = [0]
        # writer = self.writers[mode]
        for l, v in losses.items():
            wandb.log({"{}_{}".format(mode, l): v}, step=self.step)
        #     writer.add_scalar("{}".format(l), v, self.step)
        #
        # for j in range(min(4, self.opt.eval_batch_size)):  # write a maxmimum of four images
        #     for s in self.opt.scales:
        #         for frame_id in frame_ids:
        #             writer.add_image(
        #                 "color_{}_{}/{}".format(frame_id, s, j),
        #                 inputs[("color", frame_id, s)][j].data, self.step)
        #             if s == 0 and frame_id != 0:
        #                 writer.add_image(
        #                     "color_pred_{}_{}/{}".format(frame_id, s, j),
        #                     outputs[("color", frame_id, s)][j].data, self.step)
        #
        #         writer.add_image(
        #             "disp_{}/{}".format(s, j),
        #             normalize_image(outputs[("disp", s)][j]), self.step)
        #
        #         if self.opt.predictive_mask:
        #             for f_idx, frame_id in enumerate(frame_ids[1:]):
        #                 writer.add_image(
        #                     "predictive_mask_{}_{}/{}".format(frame_id, s, j),
        #                     outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
        #                     self.step)
        #
        #         elif not self.opt.disable_automasking and not val:
        #             writer.add_image(
        #                 "automask_{}/{}".format(s, j),
        #                 outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

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

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        if self.opt.beam_encoder:
            self.opt.models_to_load.append("beam_encoder")
            self.opt.models_to_load.append("beam_encoder_pose")

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    completor = Completor(opts)
    completor.train()
