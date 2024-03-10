# **Our Logit Standardization** with CTKD

This is the repository for the experiments of our logit standardization's facilitating the distillation of CTKD. It is from the [pytorch implementation](<https://github.com/zhengli97/CTKD>) for "Curriculum Temperature for Knowledge Distillation" (AAAI 2023) https://arxiv.org/abs/2211.16231


## Requirements 

- Python 3.8
- Pytorch 1.11.0
- Torchvision 0.12.0

## Running

1. Download the pretrained teacher models and put them to `./save/models`.

|  Dataset | Download |
|:---------------:|:-----------------:|
| CIFAR teacher models   | [[Baidu Yun]](https://pan.baidu.com/s/1ncvsfLTQ-GdXtKY-xtaweg?pwd=meaf)   |
| ImageNet teacher models  | [[Baidu Yun]](https://pan.baidu.com/s/1408PoziVAA8E3DojxUq1Hw?pwd=s4ma)   |

If you want to train your teacher model, please consider using `./scripts/run_cifar_vanilla.sh` or `./scripts/run_imagenet_vanilla.sh`.

After the training process, put your teacher model to `./save/models`.

2. Training on CIFAR-100:
- Download the dataset and change the path in `./dataset/cifar100.py line 27` to your current dataset path.
- Modify the script `scripts/run_cifar_distill.sh` according to your needs.
- Run the script.
    ```  bash
    sh scripts/run_cifar_distill.sh  
    ```

3. Training on ImageNet-2012:
- Download the dataset and change the path in `./dataset/imagenet.py line 21` to your current dataset path.
- Modify the script `scripts/run_imagenet_distill.sh` according to your needs.
- Run the script.
    ```  bash
    sh scripts/run_imagenet_distill.sh  
    ```

## Model Zoo
We provide complete training configs, logs, and models for your reference.

CIFAR-100:

- Combing CTKD with existing KD methods, including vanilla KD, PKT, SP, VID, CRD, SRRL, and DKD.  
(Teacher: RN-56, Student: RN-20)  
[[Baidu Yun]](https://pan.baidu.com/s/13-z-T4ooQDlWrm4isEH4qA?pwd=3bmy) [[Google]](https://drive.google.com/drive/folders/1pT8zmmOFMs5MqDLP6b4Cobv422CAcVF4?usp=sharing)

ImageNet-2012:
- Combing CTKD with vanilla KD:  
[Baidu Yun] [Google]
