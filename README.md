# Advancing Self-supervised Monocular Depth Learning with Sparse LiDAR

This paper has been accepted by [Conference on Robot Learning 2021](https://www.robot-learning.org/).

By [Ziyue Feng](https://ziyue.cool), [Longlong Jing](https://longlong-jing.github.io/), [Peng Yin](https://maxtomcmu.github.io/), [Yingli Tian](https://www.ccny.cuny.edu/profiles/yingli-tian), and [Bing Li](https://www.clemson.edu/cecas/departments/automotive-engineering/people/li.html).

Arxiv: [Link](https://arxiv.org/abs/2109.09628)

[![image](https://user-images.githubusercontent.com/21237230/136714405-de01ebac-12a6-4e5c-94bb-6ae93e4f86bf.png)](https://www.youtube.com/watch?v=_rY4ytyBQFU)

![image](https://user-images.githubusercontent.com/21237230/136713976-60e65097-5973-445a-b9f1-151ade89dcfb.png)

### Abstract
Self-supervised monocular depth prediction provides a cost-effective solution to obtain the 3D location of each pixel. However, the existing approaches usually lead to unsatisfactory accuracy, which is critical for autonomous robots. In this paper, we propose a novel two-stage network to advance the self-supervised monocular dense depth learning by leveraging low-cost sparse (e.g. 4-beam) LiDAR. Unlike the existing methods that use sparse LiDAR mainly in a manner of time-consuming iterative post-processing, our model fuses monocular image features and sparse LiDAR features to predict initial depth maps. Then, an efficient feed-forward refine network is further designed to correct the errors in these initial depth maps in pseudo-3D space with real-time performance. Extensive experiments show that our proposed model significantly outperforms all the state-of-the-art self-supervised methods, as well as the sparse-LiDAR-based methods on both self-supervised monocular depth prediction and completion tasks. With the accurate dense depth prediction, our model outperforms the state-of-the-art sparse-LiDAR-based method (Pseudo-LiDAR++) by more than 68% for the downstream task monocular 3D object detection on the KITTI Leaderboard.

## ⚙️ Setup

You can install the dependencies with:
```shell
conda create -n depth python=3.6.6
conda activate depth
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
pip install tensorboardX==1.4
conda install opencv=3.3.1   # just needed for evaluation
pip install open3d
pip install wandb
pip install scikit-image
```
We ran our experiments with PyTorch 1.8.0, CUDA 11.1, Python 3.6.6 and Ubuntu 18.04.

## 💾 KITTI Data Prepare

**Download Data**

You need to first download the KITTI RAW dataset, put in the `kitti_data` folder.

Our default settings expect that you have converted the png images to jpeg with this command, which also deletes the raw KITTI `.png` files:
```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
or you can skip this conversion step and train from raw png files by adding the flag `--png` when training, at the expense of slower load times.

**Preprocess Data**

```
bash prepare_1beam_data_for_prediction.sh
bash prepare_2beam_data_for_prediction.sh
bash prepare_3beam_data_for_prediction.sh
bash prepare_4beam_data_for_prediction.sh
bash prepare_r100.sh # random sample 100 LiDAR points
bash prepare_r200.sh # random sample 200 LiDAR points
```




## ⏳ Training

By default models and tensorboard event files are saved to `log/mdp/`.

**Depth Prediction:**

```shell
python trainer.py
python inf_depth_map.py --need_path
python inf_gdc.py
python refiner.py
```

**Depth Completion:**

Please first download the KITTI Completion dataset.
```shell
python completor.py
```

**Monocular 3D Object Detection:**

Please first download the KITTI 3D Detection dataset.

```shell
python export_detection.py
```

Then you can train the PatchNet based on the exported depth maps.


## 📊 KITTI evaluation

```shell
python evaluate_depth.py
python evaluate_completion.py
```

### Citation
```
@article{feng2021advancing,
  title={Advancing Self-supervised Monocular Depth Learning with Sparse LiDAR},
  author={Feng, Ziyue and Jing, Longlong and Yin, Peng and Tian, Yingli and Li, Bing},
  journal={arXiv preprint arXiv:2109.09628},
  year={2021}
}
```


## Reference

Our code is based on the Monodepth2: https://github.com/nianticlabs/monodepth2
