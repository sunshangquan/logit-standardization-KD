# **Our Logit Standardization** for Locality Guidance for Improving Vision Transformers on Tiny Datasets

## Description

This is the repository for the experiments of our logit standardization's facilitating the distillation of transformers. It is based on the [PyTorch implementation](<https://github.com/lkhl/tiny-transformers>) of the paper ["Locality Guidance for Improving Vision Transformers on Tiny Datasets"](<https://arxiv.org/pdf/2207.10026.pdf>), supporting different Transformer models (including DeiT, T2T-ViT, PiT, PVT, PVTv2) and the classification dataset, CIFAR-100.

## Usage

### Dependencies

The base environment we used for experiments is:

- python = 3.8.12
- pytorch = 1.8.0
- cudatoolkit = 10.1

Other dependencies can be installed by:

```shell
pip install -r requirements.txt
```

### Data Preparation

**Step 1:** download dataset from the official websites:

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

**Step 2:** move or link the datasets to `data/` directory. We show the layout of `data/` directory as follow:

```
data
└── cifar-100-python
   ├── meta
   ├── test
   └── train
```

### Train from Scratch

For example, you can train DeiT-Tiny from scratch using:

```shell
python run_net.py --mode train --cfg configs/deit/deit-ti_c100_base.yaml
```

Besides, we provide configurations for different models and different datasets at `configs/`.

### Train with Locality Guidance

**Step 1:** train the CNN guidance model (e.g., ResNet-56). This step will only take a little time and only needs to be executed once for each dataset.

```shell
python run_net.py --mode train --cfg configs/resnet/r-56_c100.yaml
```

**Step 2:** Configure ```DISTILLATION.LOGIT_STANDARD```, ```DISTILLATION.LOGIT_TEMP``` and ```DISTILLATION.EXTRA_WEIGHT_IN``` in the cfg file. Then train the target VT. 

```shell
python run_net.py --mode train --cfg configs/deit/deit-ti_c100_KD_ours.yaml

python run_net.py --mode train --cfg configs/t2t/t2t-7_c100_KD_ours.yaml

python run_net.py --mode train --cfg configs/pit/pit-ti_c100_KD_ours.yaml

python run_net.py --mode train --cfg configs/pvt/pvt-ti_c100_KD_ours.yaml
```


### Test

```shell
python run_net.py --mode test --cfg configs/deit/deit-ti_c100_base.yaml TEST.WEIGHTS /path/to/model.pyth
```

## Results

|    Model    |                      Top-1 Acc. (Base)                       |                      Top-1 Acc. (ECCV2022)                       | Tpo-1 Acc. (KD+Ours) |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  DeiT-Tiny  | 65.08 ( [weights](https://drive.google.com/file/d/1UpnIPvcTWrBZ2FYCYYY4FkTK4LhXazUY/view?usp=sharing) \| [log](https://drive.google.com/file/d/1uAIoYeNPOIE141AO-95JnKUZqKPgtz3C/view?usp=sharing) ) | 78.15 ( [weights](https://drive.google.com/file/d/1vo8jugJkgxmgFtiS4V1tIKAfmg5jdh0D/view?usp=sharing) \| [log](https://drive.google.com/file/d/1agOqk8eIGK3_XqfNnLPKOKwDbKBeqffu/view?usp=sharing) ) | 78.55( [weights](https://drive.google.com/file/d/172r35OWaXFXopjQJNy5X0JRSLTtearvv/view?usp=sharing) \| [log](../logs/tiny-transformer/deit-ti_c100_KD_ours_3_10.txt)) |
|  T2T-ViT-7  | 69.37 ( [weights](https://drive.google.com/file/d/1walDSuqyy2zfQv55NuG9a8Eq5d3GlRuf/view?usp=sharing) \| [log](https://drive.google.com/file/d/17xsso8wUlt-cf_-oZavTn9i-c-pTMhUW/view?usp=sharing) ) | 78.35 ( [weights](https://drive.google.com/file/d/1wD3wQ13O7otXjRo-4dC9DHg_HdLoUTVT/view?usp=sharing) \| [log](https://drive.google.com/file/d/1SNILqkf18lX-qcKdkg200ZBYB3N-bOue/view?usp=sharing) ) |78.43( [weights](https://drive.google.com/file/d/1W6zQvIHa9EwvJXb9dwGsgjSTBvMANfIK/view?usp=sharing) \| [log](../logs/tiny-transformer/t2t-7_c100_KD_ours_3_6.txt)) |
|  PiT-Tiny   | 73.58 ( [weights](https://drive.google.com/file/d/1bTG9W0Kf-xNJSA35xv-Wmiw6G1Bfts3m/view?usp=sharing) \| [log](https://drive.google.com/file/d/1qhRMRp-AqBSFLvspHEsM06ANf8p6STox/view?usp=sharing) ) | 78.48 ( [weights](https://drive.google.com/file/d/14dPs5CzhVKqTwuwK3n75C-SWiWa3IQ6A/view?usp=sharing) \| [log](https://drive.google.com/file/d/1zYK9i9YN2mV9GMM02nbPRMOOGwqvehJg/view?usp=sharing) ) |78.76( [weights](https://drive.google.com/file/d/1sBy44PZt0Hn-24Xh3cYEwIIThTU9lO-g/view?usp=sharing) \| [log](../logs/tiny-transformer/pit-ti_c100_KD_ours.txt)) |
|  PVT-Tiny   | 69.22 ( [weights](https://drive.google.com/file/d/18BbtQ3XF-_tzOB9BNbu04C-KDsHhrqmM/view?usp=sharing) \| [log](https://drive.google.com/file/d/1Qb3sOi0AuXl726hqxXCZSI7i-qH8_1YL/view?usp=sharing) ) | 77.07 ( [weights](https://drive.google.com/file/d/1rDFwcz3s1Irxk3FE4OhHks7qlzmoxM-w/view?usp=sharing) \| [log](https://drive.google.com/file/d/1FJ5ajTGN6zr0Eo12B8gW4XJ2FUIMSNoT/view?usp=sharing) ) |78.43( [weights](https://drive.google.com/file/d/1Ms-Vq5UEpZK1aSjQ3UZGSVJLFhGZdwLX/view?usp=sharing) \| [log](../logs/tiny-transformer/pvt-ti_c100_KD_ours.txt)) |

Here we provide pre-trained models and training logs (can be viewed via TensorBoard).

## Acknowledgement

This repository is built upon [tiny-transformers](<https://github.com/lkhl/tiny-transformers>), [pycls](https://github.com/facebookresearch/pycls) and the official implementations of [DeiT](https://github.com/facebookresearch/deit), [T2T-ViT](https://github.com/yitu-opensource/T2T-ViT), [PiT](https://github.com/naver-ai/pit), [PVTv1/v2](https://github.com/whai362/PVT), [ConViT](https://github.com/facebookresearch/convit) and [CvT](https://github.com/microsoft/CvT). We would like to thank authors of these open source repositories.