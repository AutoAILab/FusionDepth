from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default='kitti_data')
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default='log')


        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=50,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=5)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=10)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=4)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load",
                                 default='log/1337/models/weights_best/')
        self.parser.add_argument("--train_load_weights_folder",
                                 type=str,
                                 help="name of model to load",
                                 default=None)
        self.parser.add_argument("--refine_load_weights_folder",
                                 type=str,
                                 help="name of model to load",
                                 default='log/mdp/models/weights_absrel7817/')
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--eval_gdc",
                                 help="if set will perform GDC in evaluation "
                                      "from the Pseudo-Lidar paper",
                                 action="store_true")
        self.parser.add_argument("--eval_batch_size",
                                 type=int,
                                 help="batch size",
                                 default=1)
        # Concat 4 beam
        self.parser.add_argument("--need_4beam",
                                 help="load 4 beam depth map in data loader",
                                 action="store_false")
        self.parser.add_argument("--need_full_res_4beam",
                                 help="load 1242x375 4 beam depth map in data loader",
                                 action="store_true")
        self.parser.add_argument("--need_path",
                                 help="include file path in dataloader",
                                 action="store_true")
        self.parser.add_argument("--cat_4beam_to_color",
                                 help="Concat 4 beam depth map into input RGB image",
                                 action="store_true")
        self.parser.add_argument("--need_2_channel",
                                 help="Generate expanded depth and confidence map",
                                 action="store_false")
        self.parser.add_argument("--cat2start",
                                 help="Concat expanded depth and confidence map into input RGB image",
                                 action="store_true")
        self.parser.add_argument("--cat2end",
                                 help="Concat expanded depth and confidence map before output",
                                 action="store_true")
        self.parser.add_argument("--beam_encoder",
                                 help="use a separate encoder to accept 2channel input",
                                 action="store_false")
        self.parser.add_argument("--trainer_siloss",
                                 help="apply trainer_siloss",
                                 type=str,
                                 default="true",
                                 choices=["true", "false"],)
        self.parser.add_argument("--trainer_siloss_all_scale",
                                 help="trainer_siloss compute on all scale",
                                 action="store_false")
        self.parser.add_argument("--random_sample",
                                 type=int,
                                 help="num points for random sampling",
                                 default=-1)

        # Refine
        self.parser.add_argument("--train_entire_net",
                                 help="make entire network trainable",
                                 action="store_true")
        self.parser.add_argument("--refine_shallow",
                                 help="use a shallow net",
                                 action="store_true")
        self.parser.add_argument("--refineUnet",
                                 help="use a shallow net",
                                 action="store_true")
        self.parser.add_argument("--refine_deep",
                                 help="use a shallow net",
                                 action="store_true")
        self.parser.add_argument("--refine_2d",
                                 help="use a 2d refine net",
                                 action="store_true")
        self.parser.add_argument("--refine_iter",
                                 type=int,
                                 help="#iterations",
                                 default=1)
        self.parser.add_argument("--refine_iter_gama",
                                 type=float,
                                 help="exponential weights for iter loss",
                                 default=0.8)
        self.parser.add_argument("--refine_offset",
                                 help="predict a offset to the previous depth prediction",
                                 action="store_true")
        self.parser.add_argument("--refine_depthnet_with_beam",
                                 help="input beam feature to depthnet while refiner training",
                                 type=str,
                                 default="false",
                                 choices=["true", "false"])
        self.parser.add_argument("--clone_gdc",
                                 help="train a NN to clone gdc",
                                 action="store_true")
        self.parser.add_argument("--clone_path",
                                 help="model name of the clone log",
                                 type=str)
        self.parser.add_argument("--need_inf_gdc",
                                 help="load inf_depth and inf_gdc",
                                 action="store_true")
        self.parser.add_argument("--catxy",
                                 help="cat x, y coordinates to refine net input",
                                 type=str,
                                 default="true",
                                 choices=["true", "false"])
        self.parser.add_argument("--refine2d_deep",
                                 help="use a deeper refine net",
                                 type=str,
                                 default="true",
                                 choices=["true", "false"])
        self.parser.add_argument("--refine_a0",
                                 help="use the coarse depth map at scale 0 as input at all scales",
                                 type=str,
                                 default="true",
                                 choices=["true", "false"])

        self.parser.add_argument("--gdc_loss_threshold",
                                 type=float,
                                 help="gdc clone L2 loss threshold (on disparity domain)",
                                 default=2.0)
        self.parser.add_argument("--gdc_loss_weight",
                                 type=float,
                                 help="gdc clone L2 loss multiplier",
                                 default=0.008)
        self.parser.add_argument("--gdc_loss_only_on_scale_0",
                                 help="gdc clone si loss only compute on scale 0",
                                 action="store_false")
        self.parser.add_argument("--gdc_abs_loss",
                                 type=float,
                                 help="absolute loss weight on gdc cloning",
                                 default=0.0)
        self.parser.add_argument("--si_var",
                                 type=float,
                                 help="variance focus in si loss, 1 to be si loss, 0 to be l2 loss",
                                 default=0.3)


        # Completion
        self.parser.add_argument('--completion_val_split',
                                 type=str,
                                 default="select",
                                 choices=["select", "full"],
                                 help='full or select validation set')
        self.parser.add_argument("--completion_siloss_weight",
                                 type=float,
                                 help="completion si loss multiplier",
                                 default=0.1)
        self.parser.add_argument("--completion_siloss_all_scale",
                                 help="completion_siloss compute on all scale",
                                 type=str,
                                 default="false",
                                 choices=["true", "false"])
        self.parser.add_argument("--completion_eigen_crop",
                                 help="apply eigen crop in completion evaluation",
                                 action="store_true")
        self.parser.add_argument("--completion_num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=3)
        self.parser.add_argument("--completion_scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=25)
        self.parser.add_argument("--completion_not_full_res",
                                 help="apply full resolution to completion",
                                 action="store_true")
        self.parser.add_argument("--completion_amp",
                                 help="apply auto mixed precision on completor training",
                                 action="store_true")
        self.parser.add_argument("--completion_pose_num_layers",
                                 type=int,
                                 help="pose-net and beam-encoder-pose layers",
                                 default=18)
        self.parser.add_argument("--completion_siloss",
                                 help="apply siloss in completion",
                                 action="store_false")
        self.parser.add_argument("--completion_l1loss",
                                 help="apply L1 loss in completion",
                                 action="store_true")
        self.parser.add_argument("--completion_clip",
                                 type=float,
                                 help="completion gradient clip",
                                 default=0.01)
        self.parser.add_argument("--completion_num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=50,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--completion_need2channel",
                                 help="load expanded 2channel depth for completion",
                                 type=str,
                                 default="false",
                                 choices=["true", "false"], )
        self.parser.add_argument("--completion_test",
                                 help="inference on the testing dataset",
                                 action="store_true")

        # Debug
        self.parser.add_argument("--debug",
                                 help="work in debug mode, will output or visualize some data",
                                 action="store_true")

        # Visualize
        self.parser.add_argument("--visualize",
                                 help="visualize the evaluation results",
                                 action="store_true")
        self.parser.add_argument("--vis_name",
                                 help="saved error figure name",
                                 type=str,
                                 default='diff')
        self.parser.add_argument("--save_sample",
                                 type=int,
                                 help="#which sample to save, 0-696",
                                 default=-1)
        self.parser.add_argument("--inf",
                                 help="inference specified samples",
                                 action="store_true")
        self.parser.add_argument("--demo",
                                 help="inference demo samples",
                                 action="store_true")

        # Depth Guided Conv
        self.parser.add_argument("--use_dropout",
                                 help="use_dropout",
                                 type=str,
                                 default="true",
                                 choices=["true", "false"], )
        self.parser.add_argument("--drop_channel",
                                 help="drop_channel",
                                 type=str,
                                 default="true",
                                 choices=["true", "false"], )
        self.parser.add_argument("--dropout_rate",
                                 type=float,
                                 help="dropout_rate",
                                 default=0.5)
        self.parser.add_argument("--dropout_position",
                                 help="dropout_position",
                                 type=str,
                                 default="early",
                                 choices=["early", "late", "adaptive"])
        self.parser.add_argument("--base_model",
                                 type=int,
                                 help="res net layers",
                                 default=50)
        self.parser.add_argument("--adaptive_diated",
                                 help="adaptive_diated",
                                 type=str,
                                 default="true",
                                 choices=["true", "false"], )
        self.parser.add_argument("--deformable",
                                 help="use deformable conv",
                                 type=str,
                                 default="false",
                                 choices=["true", "false"], )
        self.parser.add_argument("--use_rcnn_pretrain",
                                 help="use_rcnn_pretrain",
                                 type=str,
                                 default="false",
                                 choices=["true", "false"], )
        self.parser.add_argument("--d4twocha",
                                 help="cat 2cha to input of d4lcn",
                                 type=str,
                                 default="false",
                                 choices=["true", "false"], )

        # Detection
        self.parser.add_argument("--det_name",
                                 help="output folder name",
                                 type=str)

        # Evaluation
        self.parser.add_argument("--per_semantic",
                                 help="evaluate depth per semantic category",
                                 action="store_true")
        self.parser.add_argument("--run_name",
                                 help="the saved results name",
                                 type=str)
        self.parser.add_argument('--nbeams', default=4, type=int)



    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
