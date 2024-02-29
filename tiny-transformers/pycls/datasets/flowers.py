import os
import numpy as np
from PIL import Image
from scipy.io import loadmat

from .base import BaseDataset
from pycls.core.io import pathmgr


class Flowers(BaseDataset):

    def __init__(self, data_path, split):
        super(Flowers, self).__init__(split)
        assert pathmgr.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "test"]
        assert split in splits, "Split '{}' not supported for Flowers".format(split)
        self.data_path = data_path
        self.labels = loadmat(os.path.join(data_path, 'imagelabels.mat'))['labels'][0] - 1
        all_files = loadmat(os.path.join(data_path, 'setid.mat'))
        if split == 'train':
            self.ids = np.concatenate([all_files['trnid'][0], all_files['valid'][0]])
        else:
            self.ids = all_files['tstid'][0]

    def __len__(self):
        return len(self.ids)

    def _get_data(self, idx):
        label = self.labels[self.ids[idx] - 1]
        fname = 'image_%05d.jpg'%self.ids[idx]
        img = Image.open(os.path.join(self.data_path, 'jpg', fname))
        return img, label
