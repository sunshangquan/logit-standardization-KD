from torchvision.datasets import CIFAR100

from .base import BaseDataset
from pycls.core.io import pathmgr


class Cifar100(BaseDataset):

    def __init__(self, data_path, split):
        super(Cifar100, self).__init__(split)
        assert pathmgr.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "test"]
        assert split in splits, "Split '{}' not supported for cifar".format(split)
        self.database = CIFAR100(root=data_path, train=split=='train', download=True)

    def __len__(self):
        return len(self.database)

    def _get_data(self, index):
        return self.database[index]
