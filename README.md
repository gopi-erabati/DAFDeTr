# DAFDeTr: Deformable Attention Fusion Based 3D Detection Transformer

This is the official PyTorch implementation of the paper **DAFDeTr: Deformable Attention Fusion Based 3D Detection Transformer**, by Gopi Krishna Erabati and Helder Araujo.

Gopi Krishna Erabati and Helder Araujo, ``[DAFDeTr: Deformable Attention Fusion Based 3D Detection Transformer](https://doi.org/10.1007/978-3-031-59057-3_19),'' in *Robotics, Computer Vision and Intelligent Systems (ROBOVIS 2024), Communications in Computer and Information Science*, vol 2077, pp. 293-315, 2024.

**Contents**
1. [Overview](https://github.com/gopi-erabati/DAFDeTr#overview)
2. [Results](https://github.com/gopi-erabati/DAFDeTr#results)
3. [Requirements, Installation and Usage](https://github.com/gopi-erabati/DAFDeTr#requirements-installation-and-usage)
    1. [Prerequistes](https://github.com/gopi-erabati/DAFDeTr#prerequisites)
    2. [Installation](https://github.com/gopi-erabati/DAFDeTr#installation)
    3. [Training](https://github.com/gopi-erabati/DAFDeTr#training)
    4. [Testing](https://github.com/gopi-erabati/DAFDeTr#testing)
4. [Acknowledgements](https://github.com/gopi-erabati/DAFDeTr#acknowlegements)
5. [Reference](https://github.com/gopi-erabati/DAFDeTr#reference)

## Overview
Existing approaches fuse the LiDAR points and image pixels by hard association relying on highly accurate calibration matrices. We propose Deformable Attention Fusion based 3D Detection Transformer (DAFDeTr) to attentively and adaptively fuse the image features to the LiDAR features with soft association using deformable attention mechanism. Specifically, our detection head consists of two decoders for sequential fusion: LiDAR and image decoder powered by deformable cross-attention to link the multi-modal features to the 3D object predictions leveraging a sparse set of object queries. The refined object queries from the LiDAR decoder attentively fuse with the corresponding and required image features establishing a soft association, thereby making our model robust for any camera malfunction. We conduct extensive experiments and analysis on nuScenes and Waymo datasets. Our DAFDeTr-L achieves 63.4 mAP and outperforms well established networks on the nuScenes dataset and obtains competitive performance on the Waymo dataset. Our fusion model DAFDeTr achieves 64.6 mAP on the nuScenes dataset. We also extend our model to the 3D tracking task and our model outperforms state-of-the-art methods on 3D tracking.

![dafdetr](https://github.com/gopi-erabati/DAFDeTr/assets/22390149/a327bf36-79ed-4775-9866-0b1cc71332e6)

## Results

### Predictions on nuScenes dataset
![DAFDeTr_DeformableAttentionFusionbased3DDetectionTransformer-ezgif com-video-to-gif-converter](https://github.com/gopi-erabati/DAFDeTr/assets/22390149/0b429db0-6564-48d1-9775-540e2aa0c1d6)

### nuScenes dataset
| Config | mAP | NDS | |
| :---: | :---: |:---: |:---: |
[dafdetr_res_voxel_nus_L.py](configs/dafdetr_res_voxel_nus_L.py) | 63.4 | 69.1 | [weights](https://drive.google.com/file/d/1K5Y8-c4z__AJwaF23ThKu7zqZoWSQwXy/view?usp=sharing)
[dafdetr_res_voxel_nus_LC.py](configs/dafdetr_res_voxel_nus_LC.py) | 64.6 | 69.3 | [weights](https://drive.google.com/file/d/1M3AWkHN7rUB9BzXey0hKQn5o_XBwSnaO/view?usp=sharing)

### Waymo dataset (mAPH)
| Config | Veh. L1 | Veh. L2 | Ped. L1  | Ped. L2  | Cyc. L1 | Cyc. L2 |
| :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  |
| [dafdetr_voxel_waymo_L.py](configs/dafdetr_voxel_waymo_L.py) | 71.8 | 63.5 | 61.7 | 54.2 | 66.0 | 63.5 |

We can not distribute the model weights on Waymo dataset due to the [Waymo license terms](https://waymo.com/open/terms).

## Requirements, Installation and Usage

### Prerequisites

The code is tested on the following configuration:
- Ubuntu 20.04.6 LTS
- CUDA==11.7
- Python==3.8.10
- PyTorch==1.13.1
- [mmcv](https://github.com/open-mmlab/mmcv)==1.7.0
- [mmdet](https://github.com/open-mmlab/mmdetection)==2.28.2
- [mmseg](https://github.com/open-mmlab/mmsegmentation)==0.30.0
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)==1.0.0.rc6

### Installation
```
mkvirtualenv dafdetr

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -U openmim
mim install mmcv-full==1.7.0

pip install -r requirements.txt
```
For evaluation on Waymo, please follow the below code to build the binary file `compute_detection_metrics_main` for metrics computation and put it into ```mmdet3d_plugin/core/evaluation/waymo_utils/```.
```
# download the code and enter the base directory
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
# git clone https://github.com/Abyssaledge/waymo-open-dataset-master waymo-od # if you want to use faster multi-thread version.
cd waymo-od
git checkout remotes/origin/master

# use the Bazel build system
sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=3.1.0
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential

# configure .bazelrc
./configure.sh
# delete previous bazel outputs and reset internal caches
bazel clean

bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main ../DAFDeTr/mmdet3d_plugin/core/evaluation/waymo_utils/
```

### Data
Follow [MMDetection3D-1.0.0.rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6) to prepare the [nuScenes](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html), and [Waymo](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html) datasets and symlink the data directories to `data/` folder of this repository.


**Warning:** Please strictly follow [MMDetection3D-1.0.0.rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6) code to prepare the data because other versions of MMDetection3D have different coordinate refactoring.

### Clone the repository
```
git clone https://github.com/gopi-erabati/DAFDeTr.git
cd DAFDeTr
```

### Training

#### nuScenes dataset
- Download the [backbone pretrained weights](https://drive.google.com/file/d/10zwNrqXTwdPgoIt9mvGd4LMGC3Fq9Lcr/view?usp=sharing) to `ckpts/`
- Single GPU training
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/dafdetr_res_voxel_nus_L.py --work-dir {WORK_DIR}` for LiDAR-only model and `python tools/train.py configs/dafdetr_res_voxel_nus_LC.py --work-dir {WORK_DIR} --cfg-options load_from=/path/to/lidar-only/model` for LiDAR-Camera fusion model.
- Multi GPU training
  `tools/dist_train.sh configs/dafdetr_res_voxel_nus_L.py {GPU_NUM} --work-dir {WORK_DIR}` for LiDAR-only model and `tools/dist_train.sh configs/dafdetr_res_voxel_nus_LC.py {GPU_NUM} --work-dir {WORK_DIR} --cfg-options load_from=/path/to/lidar-only/model` for LiDAR-Camera fusion model.

#### Waymo dataset 
- Single GPU training
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/dafdetr_voxel_waymo_L.py --work-dir {WORK_DIR}`
- Multi GPU training
  `tools/dist_train.sh configs/dafdetr_voxel_waymo_L.py {GPU_NUM} --work-dir {WORK_DIR}`

### Testing

#### nuScenes dataset
- Single GPU testing
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/dafdetr_res_voxel_nus_L.py /path/to/ckpt --eval mAP` for LiDAR-only model and `python tools/test.py configs/dafdetr_res_voxel_nus_LC.py /path/to/ckpt --eval mAP` for LiDAR-Camera fusion model.
- Multi GPU testing
  `tools/dist_test.sh configs/dafdetr_res_voxel_nus_L.py /path/to/ckpt {GPU_NUM} --eval mAP` for LiDAR-only model and `tools/dist_test.sh configs/dafdetr_res_voxel_nus_LC.py /path/to/ckpt {GPU_NUM} --eval mAP` for LiDAR-Camera fusion model.

#### Waymo dataset 
- Single GPU testing
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/dafdetr_voxel_waymo_L.py /path/to/ckpt --eval waymo`
- Multi GPU testing
  `tools/dist_test.sh configs/dafdetr_voxel_waymo_L.py /path/to/ckpt {GPU_NUM} --eval waymo`

## Acknowlegements
We sincerely thank the contributors for their open-source code: [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

## Reference
```
@InProceedings{10.1007/978-3-031-59057-3_19,
author="Erabati, Gopi Krishna
and Araujo, Helder",
editor="Filipe, Joaquim
and R{\"o}ning, Juha",
title="DAFDeTr: Deformable Attention Fusion Based 3D Detection Transformer",
booktitle="Robotics, Computer Vision and Intelligent Systems",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="293--315",
isbn="978-3-031-59057-3"
}

```
