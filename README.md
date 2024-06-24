# [CVPR 2024 Highlight] Logit Standardization in Knowledge Distillation

[![](https://img.shields.io/badge/project-page-red.svg)](https://sunsean21.github.io/logit-stand-KD.html) [![](https://img.shields.io/badge/arXiv-2403.01427-green.svg)](https://arxiv.org/abs/2403.01427)     [![](https://img.shields.io/badge/supplementary-material-purple)](https://sunsean21.github.io/resources/cvpr2024_supp.pdf)     [![](https://img.shields.io/badge/chinese_blog-zhihu-blue.svg)](https://zhuanlan.zhihu.com/p/688903391) [![Closed Issues](https://img.shields.io/github/issues-closed/sunshangquan/logit-standardization-KD)](https://github.com/sunshangquan/logit-standardization-KD/issues?q=is%3Aissue+is%3Aclosed) [![Open Issues](https://img.shields.io/github/issues/sunshangquan/logit-standardization-KD)](https://github.com/sunshangquan/logit-standardization-KD/issues) 

### [[Project Page]](<https://sunsean21.github.io/logit-stand-KD.html>) [[arXiv]](<https://arxiv.org/abs/2403.01427>) [[Supplementary Materials]](<https://sunsean21.github.io/resources/cvpr2024_supp.pdf>) [[Zhihu (in Chinese)]](<https://zhuanlan.zhihu.com/p/688903391>) 

<img src=.github/1_1-1.png width=50% />  |  <img src=.github/2_2-1.png width=50% />
:-------------------------:|:-------------------------:
Vanilla KD             |  KD w/ our logit standardization

## Abstract

Knowledge distillation involves transferring soft labels from a teacher to a student using a shared temperature-based softmax function. However, the assumption of a shared temperature between teacher and student implies a mandatory exact match between their logits in terms of logit range and variance. This side-effect limits the performance of student, considering the capacity discrepancy between them and the finding that the innate logit relations of teacher are sufficient for student to learn. To address this issue, we propose setting the temperature as the weighted standard deviation of logit and performing a plug-and-play Z-score pre-process of logit standardization before applying softmax and Kullback-Leibler divergence. Our pre-process enables student to focus on essential logit relations from teacher rather than requiring a magnitude match, and can improve the performance of existing logit-based distillation methods. We also show a typical case where the conventional setting of sharing temperature between teacher and student cannot reliably yield the authentic distillation evaluation; nonetheless, this challenge is successfully alleviated by our Z-score. We extensively evaluate our method for various student and teacher models on CIFAR-100 and ImageNet, showing its significant superiority. The vanilla knowledge distillation powered by our pre-process can achieve favorable performance against state-of-the-art methods, and other distillation variants can obtain considerable gain with the assistance of our pre-process.

## :tada: News 

2024.4: Selected as Highlight in CVPR 2024 

2024.3: Release the code and arXiv

2024.2: Accepted by CVPR 2024 

2023.7: Rejected by ICCV 2023 

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

- Download the [`cifar_teachers.tar`](<https://github.com/megvii-research/mdistiller/releases/tag/checkpoints>) and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.


1. For KD

  ```bash
  # KD
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml
  # KD+Ours
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```

2. For DKD

  ```bash
  # DKD
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml 
  # DKD+Ours
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```
3. For MLKD

  ```bash
  # MLKD
  python tools/train.py --cfg configs/cifar100/mlkd/resnet32x4_resnet8x4.yaml
  # MLKD+Ours
  python tools/train.py --cfg configs/cifar100/mlkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```

4. For CTKD

Please refer to [CTKD](./CTKD).

#### Results and Logs

We put the training logs in `./logs` and hyper-linked below. The name of each log file is formated with `KD_TYPE,TEACHER,STUDENT,BASE_TEMPERATURE,KD_WEIGHT.txt`. The possible third value for DKD is the value of BETA. Due to average operation and randomness, there may be slight differences between the reported results and the logged results. 

1. Teacher and student have **identical** structures:

| Teacher <br> Student |ResNet32x4 <br> ResNet8x4|VGG13 <br> VGG8|Wrn_40_2 <br> Wrn_40_1|Wrn_40_2 <br> Wrn_16_2|ResNet56 <br> ResNet20|ResNet110 <br> ResNet32|ResNet110 <br> ResNet20|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|:--------------------:|
| KD | 73.33 | 72.98 | 73.54 | 74.92 | 70.66 | 73.08 | 70.67 |
| KD+**Ours** | [76.62](<./logs/KD/kd,resnet32x4,resnet8x4,2,9.txt>) | [74.36](<logs/KD/kd,vgg13,vgg8,3,9.txt>) | [74.37](<logs/KD/kd,wrn_40_2,wrn_40_1,2,9.txt>) | [76.11](<logs/KD/kd,wrn_40_2,wrn_16_2,2,9.txt>) | [71.43](<logs/KD/kd,resnet56,resnet20,2,9.txt>) | [74.17](<logs/KD/kd,resnet110,resnet32,2,9.txt>) | [71.48](<logs/KD/kd,resnet110,resnet20,4,12.txt>) | 
| CTKD | 73.39 | 73.52 | 73.93 | 75.45 | 71.19 | 73.52 | 70.99 |
| CTKD+**Ours** | [76.67](<logs/CTKD/ctkd,resnet32x4,resnet8x4,2,9.txt>) | [74.47](<logs/CTKD/ctkd,vgg13,vgg8,2,9.txt>) | [74.58](<ogs/CTKD/ctkd,wrn_40_2,wrn_40_1,2,9.txt>) | [76.08](<logs/CTKD/ctkd,wrn_40_2,wrn_16_2,2,9.txt>) | [71.34](<logs/CTKD/ctkd,resnet56,resnet20,2,9.txt>) | [74.01](<logs/CTKD/ctkd,resnet110,resnet32,2,9.txt>) | [71.39](<logs/CTKD/ctkd,resnet110,resnet20,2,9.txt>) |
| DKD | 76.32 | 74.68 | 74.81 | 76.24 | 71.97 | 74.11 | 71.06 |
| DKD+**Ours** | [77.01](<./logs/DKD/dkd,resnet32x4,resnet8x4,2,9.txt>) | [74.81](<logs/DKD/dkd,vgg13,vgg8,2,15,2.0.txt>) | [74.89](<logs/DKD/dkd,wrn_40_2,wrn_40_1,2,12,2.0.txt>) | [76.39](<logs/DKD/dkd,wrn_40_2,wrn_16_2,2,12,2.0.txt>) | [72.32](<logs/DKD/dkd,resnet56,resnet20,3,18,2.0.txt>) | [74.29](<logs/DKD/dkd,resnet110,resnet32,2,15,2.0.txt>) | [71.85](<logs/DKD/dkd,resnet110,resnet20,2,15,2.0.txt>) |
| MLKD | 77.08 | 75.18 | 75.35 | 76.63 | 72.19 | 74.11 | 71.89 |
| MLKD+**Ours** | [**78.28**](<logs/MLKD/mlkd,resnet32x4,resnet8x4,2,9.txt>) | [**75.22**](<logs/MLKD/mlkd,vgg13,vgg8,2,6.txt>) | [**75.56**](<logs/MLKD/mlkd,wrn_40_2,wrn_40_1,2,9.txt>) | [**76.95**](<logs/MLKD/mlkd,wrn_40_2,wrn_16_2,2,9.txt>) | [**72.33**]() | [**74.32**](<logs/MLKD/mlkd,res110,res32,2,9.txt>) | [**72.27**](<logs/MLKD/mlkd,res110,res20,2,9.txt>) |

2. Teacher and student have **distinct** structures:

|Teacher <br> Student | ResNet32x4 <br> SHN-V2 | ResNet32x4 <br> Wrn_16_2 | ResNet32x4 <br> Wrn_40_2 | Wrn_40_2 <br> ResNet8x4 | Wrn_40_2 <br> MN-V2 | VGG13 <br> MN-V2 | ResNet50 <br> MN-V2 |
|:-------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|:--------------------:|
| KD | 74.45 | 74.90 | 77.70 | 73.97 | 68.36 | 67.37 | 67.35 | 
| KD+**Ours** | [75.56](<logs/KD/kd,resnet32x4,ShuffleV2,2,9.txt>) | [75.26](<logs/KD/kd,resnet32x4,wrn_16_2,3,9.txt>) | [77.92](<logs/KD/kd,resnet32x4,wrn_40_2,3,9.txt>) | [77.11](<logs/KD/kd,wrn_40_2,resnet8x4,2,9.txt>) | [69.23](<logs/KD/kd,wrn_40_2,MobileNetV2,3,9.txt>) | [68.61](<logs/KD/kd,vgg13,MobileNetV2,3,9.txt>) | [69.02](<logs/KD/kd,ResNet50,MobileNetV2,3,1.txt>) |
| CTKD | 75.37 | 74.57 | 77.66 | 74.61 | 68.34 | 68.50 | 68.67 | 
| CTKD+**Ours** | [76.18](<logs/CTKD/ctkd,resnet32x4,ShuffleV2,2,9.txt>) | [75.16](<logs/CTKD/ctkd,resnet32x4,wrn_16_2,2,9.txt>) | [77.99](<logs/CTKD/ctkd,resnet32x4,wrn_40_2,2,9.txt>) | [77.03](<logs/CTKD/ctkd,wrn_40_2,resnet8x4,2,9.txt>) | [69.53](<logs/CTKD/ctkd,wrn_40_2,MobileNetV2,2,9.txt>) | [68.98](<logs/CTKD/ctkd,vgg13,MobileNetV2,2,9.txt>) | [69.36](<logs/CTKD/ctkd,ResNet50,MobileNetV2,2,9.txt>)
| DKD | 77.07 | 75.70 | 78.46 | 75.56 | 69.28 | 69.71 | 70.35 | 
| DKD+**Ours** | [77.37](<logs/DKD/dkd,resnet32x4,ShuffleV2,2,9.txt>) | [76.19](<logs/DKD/dkd,resnet32x4,wrn_16_2,2,9,2.0.txt>) | [78.95](<logs/DKD/dkd,resnet32x4,wrn_40_2,2,9,8.0.txt>) | [76.75](<logs/DKD/dkd,wrn_40_2,resnet8x4,2,18,2.0.txt>) | [70.01](<logs/DKD/dkd,wrn_40_2,MobileNetV2,2,15,2.0.txt>) | [69.98](<logs/DKD/dkd,vgg13,MobileNetV2,3,9,2.0.txt>) | [70.45](<logs/DKD/dkd,ResNet50,MobileNetV2,2,15,2.0.txt>) |
| MLKD | 78.44 | 76.52 | 79.26 | 77.33 | 70.78 | 70.57 | 71.04 | 
| MLKD+**Ours** | [**78.76**](<logs/MLKD/mlkd,resnet32x4,ShuffleV2,2,6.txt>) | [**77.53**](<logs/MLKD/mlkd,resnet32x4,wrn_16_2,2,9.txt>) | [**79.66**](<logs/MLKD/mlkd,resnet32x4,wrn_40_2,2,9.txt>) | [**77.68**](<logs/MLKD/mlkd,wrn_40_2,resnet8x4,2,9.txt>) | [**71.61**]() | [**70.94**](<logs/MLKD/mlkd,vgg13,MobileNetV2,2,6.txt>) | [**71.19**](<logs/MLKD/mlkd,res50,mv2,2,9.txt>) |

### Training on ImageNet

- Download the dataset at <https://image-net.org/> and put it to `./data/imagenet`

  ```bash
  # KD
  python tools/train.py --cfg configs/imagenet/r34_r18/kd.yaml
  # KD+Ours
  python tools/train.py --cfg configs/imagenet/r34_r18/kd.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```

## Distilling ViTs

Please refer to [tiny-transformers](./tiny-transformers).

### Results

|    Model    |                      Top-1 Acc. (Base)                       |                      Top-1 Acc. (ECCV2022)                       | Tpo-1 Acc. (KD+Ours) |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  DeiT-Tiny  | 65.08 ( [weights](https://drive.google.com/file/d/1UpnIPvcTWrBZ2FYCYYY4FkTK4LhXazUY/view?usp=sharing) \| [log](https://drive.google.com/file/d/1uAIoYeNPOIE141AO-95JnKUZqKPgtz3C/view?usp=sharing) ) | 78.15 ( [weights](https://drive.google.com/file/d/1vo8jugJkgxmgFtiS4V1tIKAfmg5jdh0D/view?usp=sharing) \| [log](https://drive.google.com/file/d/1agOqk8eIGK3_XqfNnLPKOKwDbKBeqffu/view?usp=sharing) ) | 78.55( [weights](https://drive.google.com/file/d/172r35OWaXFXopjQJNy5X0JRSLTtearvv/view?usp=sharing) \| [log](../logs/tiny-transformer/deit-ti_c100_KD_ours_3_10.txt)) |
|  T2T-ViT-7  | 69.37 ( [weights](https://drive.google.com/file/d/1walDSuqyy2zfQv55NuG9a8Eq5d3GlRuf/view?usp=sharing) \| [log](https://drive.google.com/file/d/17xsso8wUlt-cf_-oZavTn9i-c-pTMhUW/view?usp=sharing) ) | 78.35 ( [weights](https://drive.google.com/file/d/1wD3wQ13O7otXjRo-4dC9DHg_HdLoUTVT/view?usp=sharing) \| [log](https://drive.google.com/file/d/1SNILqkf18lX-qcKdkg200ZBYB3N-bOue/view?usp=sharing) ) |78.43( [weights](https://drive.google.com/file/d/1W6zQvIHa9EwvJXb9dwGsgjSTBvMANfIK/view?usp=sharing) \| [log](../logs/tiny-transformer/t2t-7_c100_KD_ours_3_6.txt)) |
|  PiT-Tiny   | 73.58 ( [weights](https://drive.google.com/file/d/1bTG9W0Kf-xNJSA35xv-Wmiw6G1Bfts3m/view?usp=sharing) \| [log](https://drive.google.com/file/d/1qhRMRp-AqBSFLvspHEsM06ANf8p6STox/view?usp=sharing) ) | 78.48 ( [weights](https://drive.google.com/file/d/14dPs5CzhVKqTwuwK3n75C-SWiWa3IQ6A/view?usp=sharing) \| [log](https://drive.google.com/file/d/1zYK9i9YN2mV9GMM02nbPRMOOGwqvehJg/view?usp=sharing) ) |78.76( [weights](https://drive.google.com/file/d/1sBy44PZt0Hn-24Xh3cYEwIIThTU9lO-g/view?usp=sharing) \| [log](../logs/tiny-transformer/pit-ti_c100_KD_ours.txt)) |
|  PVT-Tiny   | 69.22 ( [weights](https://drive.google.com/file/d/18BbtQ3XF-_tzOB9BNbu04C-KDsHhrqmM/view?usp=sharing) \| [log](https://drive.google.com/file/d/1Qb3sOi0AuXl726hqxXCZSI7i-qH8_1YL/view?usp=sharing) ) | 77.07 ( [weights](https://drive.google.com/file/d/1rDFwcz3s1Irxk3FE4OhHks7qlzmoxM-w/view?usp=sharing) \| [log](https://drive.google.com/file/d/1FJ5ajTGN6zr0Eo12B8gW4XJ2FUIMSNoT/view?usp=sharing) ) |78.43( [weights](https://drive.google.com/file/d/1Ms-Vq5UEpZK1aSjQ3UZGSVJLFhGZdwLX/view?usp=sharing) \| [log](../logs/tiny-transformer/pvt-ti_c100_KD_ours.txt)) |

## Reproduce Figure 3, 4, 5
Please refer to [visualizations](./tools/visualizations).

# Acknowledgement
Sincere gratitude to the contributors of mdistiller, CTKD, Multi-Level-Logit-Distillation and tiny-transformers for their distinguished efforts.

# :mailbox_with_mail: Contact
Shangquan Sun: shangquansun@gmail.com

# :mega: Citation

If you find that this project helps your research, please consider citing some of the following paper:

```BibTeX
@inproceedings{sun2024logit,
  title={Logit standardization in knowledge distillation},
  author={Sun, Shangquan and Ren, Wenqi and Li, Jingzhi and Wang, Rui and Cao, Xiaochun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15731--15740},
  year={2024}
}

@article{sun2024logit,
  title={Logit Standardization in Knowledge Distillation},
  author={Sun, Shangquan and Ren, Wenqi and Li, Jingzhi and Wang, Rui and Cao, Xiaochun},
  journal={arXiv preprint arXiv:2403.01427},
  year={2024}
}
```
