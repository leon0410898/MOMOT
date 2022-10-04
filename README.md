# MOMOT: Motion-Guided Attention for Multiple Object Tracking

## Introduction
![image](https://github.com/leon0410898/MOMOT/blob/main/figs/demo.gif)

**Abstract.** The motion-guided attention is based on track queries, which are regarded as both motion prior and temporal offset predictors without specifying any extra embedding to predict motion. The motion prior explicitly indicates queries where to attend the possible keys. Thus, track queries only need to interact with a constant number of keys around the queries to save computation. This motion-guided attention is fully integrated with the transformer, which enables an end-to-end architecture. The simulation results on MOT17 have shown the state-ofthe-art performance
## Main Results

### MOT17

| **Method** | **Dataset** |    **Train Data**    | **IDF1** | **MT** | **ML** | **MOTA** | **IDF1** | **IDS** |                                           **URL**                                           |
| :--------: | :---------: | :------------------: | :------: | :----: | :----: | :------: | :------: | :-----: | :-----------------------------------------------------------------------------------------: |
|    MOMOT   |    MOT17    |    MOT17+CrowdHuman  |   65.7   |  40.3  |  19.9  |   72.8   |   65.7   |  2586   | [model](https://drive.google.com/file/d/1K5Im9tmRNGivJz7ynLEejGQhK_Ec-9fj/view?usp=sharing) |


*Note:*

1. MOMOT on MOT17is trained on 4 NVIDIA A100 GPUs.
2. The training time for MOT17 is about 1 days on A100;
3. The inference speed is about 7.2 FPS for resolution 1536x800;
4. All models of MOMOT are trained with ResNet50 with pre-trained weights on COCO dataset.


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

### Requirements

* Linux, CUDA>=11.1, GCC>=5.4

* PyTorch>=1.10.1, torchvision>=0.11.2 (following instructions [here](https://pytorch.org/))
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage

### Dataset preparation

1. Please download [MOT17 dataset](https://motchallenge.net/) and [CrowdHuman dataset](https://www.crowdhuman.org/) and organize them like [FairMOT](https://github.com/ifzhang/FairMOT) as following:

```
.
├── crowdhuman
│   ├── images
│   └── labels_with_ids
├── MOT15
│   ├── images
│   ├── labels_with_ids
│   ├── test
│   └── train
├── MOT17
│   ├── images
│   ├── labels_with_ids
├── DanceTrack
│   ├── train
│   ├── test
├── bdd100k
│   ├── images
│       ├── track
│           ├── train
│           ├── val
│   ├── labels
│       ├── track
│           ├── train
│           ├── val

```

### Training and Evaluation

#### Training on single node

You can download COCO pretrained weights from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). Then training MOMOT as following:

```bash 
sh config/momot_train.sh

```

#### Evaluation on MOT17

```bash 
sh config/momot_eval.sh

```

#### Inference on MOT17

```bash
sh config/momot_submit.sh

```
