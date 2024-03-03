# Code release for Logit Standardization in Knowledge Distillation (CVPR 2024).


<img src=.github/1_1-1.png width=50% />  |  <img src=.github/2_2-1.png width=50% />
:-------------------------:|:-------------------------:
Vanilla KD             |  KD w/ our logit standardization

## Abstract

Knowledge distillation involves transferring soft labels from a teacher to a student using a shared temperature-based softmax function. However, the assumption of a shared temperature between teacher and student implies a mandatory exact match between their logits in terms of logit range and variance. This side-effect limits the performance of student, considering the capacity discrepancy between them and the finding that the innate logit relations of teacher are sufficient for student to learn. To address this issue, we propose setting the temperature as the weighted standard deviation of logit and performing a plug-and-play Z-score pre-process of logit standardization before applying softmax and Kullback-Leibler divergence. Our pre-process enables student to focus on essential logit relations from teacher rather than requiring a magnitude match, and can improve the performance of existing logit-based distillation methods. We also show a typical case where the conventional setting of sharing temperature between teacher and student cannot reliably yield the authentic distillation evaluation; nonetheless, this challenge is successfully alleviated by our Z-score. We extensively evaluate our method for various student and teacher models on CIFAR-100 and ImageNet, showing its significant superiority. The vanilla knowledge distillation powered by our pre-process can achieve favorable performance against state-of-the-art methods, and other distillation variants can obtain considerable gain with the assistance of our pre-process.

## Usage

The code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>), [Multi-Level-Logit-Distillation](<https://github.com/Jin-Ying/Multi-Level-Logit-Distillation>), [CTKD](<https://github.com/zhengli97/CTKD>) and [tiny-transformers](<https://github.com/lkhl/tiny-transformers>).

**The repository is still in progress.**

### Installation

Environments:

- Python 3.8
- PyTorch 1.7.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python setup.py develop
```

## Distilling CNNs

### CIFAR-100

- Download the [`cifar_teachers.tar`](<https://github.com/megvii-research/mdistiller/releases/tag/checkpoints>) and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.


1. For KD

  ```bash
  # KD
  python tools/train.py --cfg configs/cifar100/KD/res32x4_res8x4.yaml
  # KD+Ours
  python tools/train.py --cfg configs/cifar100/KD/res32x4_res8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```

2. For DKD

  ```bash
  # DKD
  python tools/train.py --cfg configs/cifar100/DKD/res32x4_res8x4.yaml 
  # DKD+Ours
  python tools/train.py --cfg configs/cifar100/DKD/res32x4_res8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```
3. For MLKD

  ```bash
  # MLKD
  python tools/train.py --cfg configs/cifar100/MLKD/res32x4_res8x4.yaml
  # MLKD+Ours
  python tools/train.py --cfg configs/cifar100/MLKD/res32x4_res8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```

4. For CTKD

Please refer to [CTKD](./CTKD) (in progress).

#### Results and Logs

We put the training logs in `./logs` and hyper-linked below. The name of each log file is formated with `KD_TYPE,TEACHER,STUDENT,BASE_TEMPERATURE,KD_WEIGHT.txt`. Due to average operation and randomness, there may be slight differences between the reported results and the logged results. 

1. Teacher and student have **identical** structures:

| Teacher <br> Student |ResNet32x4 <br> ResNet8x4|VGG13 <br> VGG8|Wrn_40_2 <br> Wrn_40_1|Wrn_40_2 <br> Wrn_16_2|ResNet56 <br> ResNet20|ResNet110 <br> ResNet32|ResNet110 <br> ResNet20|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|:--------------------:|
| KD | 73.33 | 72.98 | 73.54 | 74.92 | 70.66 | 73.08 | 70.67 |
| KD+**Ours** | [76.62](<./logs/KD/kd,resnet32x4,resnet8x4,2,9.txt>) | [74.36](<logs/KD/kd,vgg13,vgg8,3,9.txt>) | [74.37](<logs/KD/kd,wrn_40_2,wrn_40_1,2,9.txt>) | [76.11](<logs/KD/kd,wrn_40_2,wrn_16_2,2,9.txt>) | [71.43](<logs/KD/kd,resnet56,resnet20,2,9.txt>) | [74.17](<logs/KD/kd,resnet110,resnet32,2,9.txt>) | 71.48 | 
| CTKD | 73.39 | 73.52 | 73.93 | 75.45 | 71.19 | 73.52 | 70.99 |
| CTKD+**Ours** | [76.67](<logs/CTKD/ctkd,resnet32x4,resnet8x4,2,9.txt>) | [74.47](<logs/CTKD/ctkd,vgg13,vgg8,2,9.txt>) | [74.58](<ogs/CTKD/ctkd,wrn_40_2,wrn_40_1,2,9.txt>) | [76.08](<logs/CTKD/ctkd,wrn_40_2,wrn_16_2,2,9.txt>) | [71.34](<logs/CTKD/ctkd,resnet56,resnet20,2,9.txt>) | 74.01 | 71.39 |
| DKD | 76.32 | 74.68 | 74.81 | 76.24 | 71.97 | 74.11 | 71.06 |
| DKD+**Ours** | [77.01](<./logs/DKD/dkd,resnet32x4,resnet8x4,2,9.txt>) | 74.81 | 74.89 | 76.39 | 72.32 | 74.29 | 71.85 |
| MLKD | 77.08 | 75.18 | 75.35 | 76.63 | 72.19 | 74.11 | 71.89 |
| MLKD+**Ours** | [**78.28**](<logs/MLKD/mlkd,resnet32x4,resnet8x4,2,9.txt>) | **75.22** | [**75.56**](<logs/MLKD/mlkd,wrn_40_2,wrn_40_1,2,9.txt>) | **76.95** | **72.33** | [**74.32**](<logs/MLKD/mlkd,res110,res32,2,9.txt>) | [**72.27**](<logs/MLKD/mlkd,res110,res20,2,9.txt>) |

2. Teacher and student have **distinct** structures:

|Teacher <br> Student | ResNet32x4 <br> SHN-V2 | ResNet32x4 <br> Wrn_16_2 | ResNet32x4 <br> Wrn_40_2 | Wrn_40_2 <br> ResNet8x4 | Wrn_40_2 <br> MN-V2 | VGG13 <br> MN-V2 | ResNet50 <br> MN-V2 |
|:-------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|:--------------------:|
| KD | 74.45 | 74.90 | 77.70 | 73.97 | 68.36 | 67.37 | 67.35 | 
| KD+**Ours** | [75.56](<logs/KD/kd,resnet32x4,ShuffleV2,2,9.txt>) | [75.26](<logs/KD/kd,resnet32x4,wrn_16_2,3,9.txt>) | [77.92](<logs/KD/kd,resnet32x4,wrn_40_2,3,9.txt>) | [77.11](<logs/KD/kd,wrn_40_2,resnet8x4,2,9.txt>) | [69.23](<logs/KD/kd,wrn_40_2,MobileNetV2,3,9.txt>) | [68.61](<logs/KD/kd,vgg13,MobileNetV2,3,9.txt>) | [69.02](<logs/KD/kd,ResNet50,MobileNetV2,3,1.txt>) |
| CTKD | 75.37 | 74.57 | 77.66 | 74.61 | 68.34 | 68.50 | 68.67 | 
| CTKD+**Ours** | [76.18](<logs/CTKD/ctkd,resnet32x4,ShuffleV2,2,9.txt>) | [75.16](<logs/CTKD/ctkd,resnet32x4,wrn_16_2,2,9.txt>) | [77.99](<logs/CTKD/ctkd,resnet32x4,wrn_40_2,2,9.txt>) | [77.03](<logs/CTKD/ctkd,wrn_40_2,resnet8x4,2,9.txt>) | [69.53](<logs/CTKD/ctkd,wrn_40_2,MobileNetV2,2,9.txt>) | [68.98](<logs/CTKD/ctkd,vgg13,MobileNetV2,2,9.txt>) | 69.36
| DKD | 77.07 | 75.70 | 78.46 | 75.56 | 69.28 | 69.71 | 70.35 | 
| DKD+**Ours** | [77.37](<logs/DKD/dkd,resnet32x4,ShuffleV2,2,9.txt>) | 76.19 | 78.95 | 76.75 | 70.01 | 69.98 | 70.45 |
| MLKD | 78.44 | 76.52 | 79.26 | 77.33 | 70.78 | 70.57 | 71.04 | 
| MLKD+**Ours** | **78.76** | [**77.53**](<logs/MLKD/mlkd,resnet32x4,wrn_16_2,2,9.txt>) | [**79.66**](<logs/MLKD/mlkd,resnet32x4,wrn_40_2,2,9.txt>) | **77.68** | **71.61** | **70.94** | [**71.19**](<logs/MLKD/mlkd,res50,mv2,2,9.txt>) |

### Training on ImageNet

- Download the dataset at <https://image-net.org/> and put it to `./data/imagenet`

  ```bash
  python tools/train.py --cfg configs/imagenet/r34_r18/kd_ours.yaml
  ```

## Distilling ViTs

Please refer to [tiny-transformers](./tiny-transformers) (in progress).


# Acknowledgement
Sincere gratitude to the contributors of mdistiller, CTKD, Multi-Level-Logit-Distillation and tiny-transformers for their distinguished efforts.

# Contact
[Shangquan Sun](https://sunsean21.github.io/): shangquansun@gmail.com

# Citation

If you find that this project helps your research, please consider citing some of the following paper:

```BibTeX
@inproceedings{Sun2024Logit,
    title={Logit Standardization in Knowledge Distillation},
    author={Shangquan Sun, Wenqi Ren, Jingzhi Li, Rui Wang, Xiaochun Cao},
    booktitle={Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```