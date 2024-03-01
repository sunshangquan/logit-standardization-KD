# Code release for Logit Standardization in Knowledge Distillation (CVPR 2024).


<img src=.github/1_1-1.png width=60% />  |  <img src=.github/2_2-1.png width=60% />
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
sudo python3 setup.py develop
```

## Distilling CNNs

### CIFAR-100

- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.


  ```bash
  python3 tools/train_ours.py --cfg configs/cifar100/ours/res32x4_res8x4.yaml 
  ```

### Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`

  ```bash
  python3 tools/train_ours.py --cfg configs/imagenet/r34_r18/kd_ours.yaml
  ```

## Distilling ViTs

Please refer to [tiny-transformers/README.md](./tiny-transformers/README.md)


# Acknowledgement
Sincere gratitude to the contributors of mdistiller, CTKD, Multi-Level-Logit-Distillation and tiny-transformers for their distinguished efforts.

# Contact
[Shangquan Sun](https://sunsean21.github.io/): shangquansun@gmail.com
