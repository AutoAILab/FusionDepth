# Advancing Self-supervised Monocular Depth Learning with Sparse LiDAR

This is the PyTorch implementation for training and testing of our model in CoRL submission #278

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



## Reference

Our code is based on the Monodepth2: https://github.com/nianticlabs/monodepth2