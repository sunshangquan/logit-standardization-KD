# Code release for Logit Standardization in Knowledge Distillation (CVPR 2024).


<img src=.github/1_1-1.png width=40% />  |  <img src=.github/2_2-1.png width=40% />
:-------------------------:|:-------------------------:
Vanilla KD             |  KD w/ our logit standardization

## Abstract

Knowledge distillation involves transferring soft labels from a teacher to a student using a shared temperature-based softmax function. However, the assumption of a shared temperature between teacher and student implies a mandatory exact match between their logits in terms of logit range and variance. This side-effect limits the performance of student, considering the capacity discrepancy between them and the finding that the innate logit relations of teacher are sufficient for student to learn. To address this issue, we propose setting the temperature as the weighted standard deviation of logit and performing a plug-and-play Z-score pre-process of logit standardization before applying softmax and Kullback-Leibler divergence. Our pre-process enables student to focus on essential logit relations from teacher rather than requiring a magnitude match, and can improve the performance of existing logit-based distillation methods. We also show a typical case where the conventional setting of sharing temperature between teacher and student cannot reliably yield the authentic distillation evaluation; nonetheless, this challenge is successfully alleviated by our Z-score. We extensively evaluate our method for various student and teacher models on CIFAR-100 and ImageNet, showing its significant superiority. The vanilla knowledge distillation powered by our pre-process can achieve favorable performance against state-of-the-art methods, and other distillation variants can obtain considerable gain with the assistance of our pre-process.

## Usage

The code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>), [Multi-Level-Logit-Distillation](<https://github.com/Jin-Ying/Multi-Level-Logit-Distillation>), [CTKD](<https://github.com/zhengli97/CTKD>) and [tiny-transformers](<https://github.com/lkhl/tiny-transformers>).

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

- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.


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

Please refer to [CTKD/README.md](./CTKD/README.md)

#### Results and Logs

| Teacher <br> Student |ResNet32x4 <br> ResNet8x4|VGG13 <br> VGG8|Wrn-40-2 <br> Wrn-40-1|Wrn-40-2 <br> Wrn-16-2|ResNet56 <br> ResNet20|ResNet110 <br> ResNet32|ResNet110 <br> ResNet20|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|:--------------------:|
| KD | 73.33 | 72.98 | 73.54 | 74.92 | 70.66 | 73.08 | 70.67 |
| KD+Ours | [76.62](<./logs/KD/kd,resnet32x4,resnet8x4,2,9.txt>) | 74.36 | 74.37 | 76.11 | 71.43 | 74.17 | 71.48 | 
| CTKD | 73.39 | 73.52 | 73.93 | 75.45 | 71.19 | 73.52 | 70.99 |
| CTKD+Ours | 76.67 | 74.47 | 74.58 | 76.08 | 71.34 | 74.01 | 71.39 |
| DKD | 76.32 | 74.68 | 74.81 | 76.24 | 71.97 | 74.11 | 71.06 |
| DKD+Ours | [77.01](<./logs/DKD/dkd,resnet32x4,resnet8x4,2,9.txt>) | 74.81 | 74.89 | 76.39 | 72.32 | 74.29 | 71.85 |
| MLKD | 77.08 | 75.18 | 75.35 | 76.63 | 72.19 | 74.11 | 71.89 |
| **MLKD+Ours** | **78.28** | **75.22** | **75.56** | **76.95** | **72.33** | **74.32** | **72.27** |

### Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`

  ```bash
  python tools/train.py --cfg configs/imagenet/r34_r18/kd_ours.yaml
  ```

## Distilling ViTs

Please refer to [tiny-transformers/README.md](./tiny-transformers/README.md)


# Acknowledgement
Sincere gratitude to the contributors of mdistiller, CTKD, Multi-Level-Logit-Distillation and tiny-transformers for their distinguished efforts.

# Contact
[Shangquan Sun](https://sunsean21.github.io/): shangquansun@gmail.com
