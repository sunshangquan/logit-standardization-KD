import argparse
import pycls.core.config as config
import pycls.core.builders as builders
from pycls.datasets.transforms import create_test_transform

import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/resnet/r-56_c100.yaml')
    parser.add_argument('--ckpt', type=str, default='work_dirs/r-56_c100/model.pyth')
    args = parser.parse_args()

    save_dir = 'temp'
    os.makedirs(save_dir, exist_ok=True)

    config.load_cfg(args.cfg)

    transforms = create_test_transform()
    transform = Compose(transforms)
    dataset = CIFAR100(root='data', train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=200)

    model = builders.build_model()
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model_state'])
    model.cuda()

    feat_dict = defaultdict(list)
    for img, _ in tqdm(loader):
        img = img.cuda()
        with torch.no_grad():
            model(img)

        for i, feat in enumerate(model.features):
            N, _, H, W = feat.shape
            feat = feat.cpu()
            feat_dict[f'layer_{i}'].append(feat)

    for k in feat_dict:
        feat_dict[k] = torch.cat(feat_dict[k], dim=0).numpy()

    cfg_name = os.path.splitext(os.path.basename(args.cfg))[0]
    np.savez(os.path.join(save_dir, f'{cfg_name}.npz'), **feat_dict)


if __name__ == '__main__':
    main()
