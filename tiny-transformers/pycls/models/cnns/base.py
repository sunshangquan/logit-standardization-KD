import torch.nn as nn
from abc import ABCMeta, abstractmethod

from pycls.core.config import cfg


class BaseConvModel(nn.Module, metaclass=ABCMeta):
    """
    Base class for conv models.

    Attributes:
        - self.features (List[Tensor]): the features in each stage.
        - self.feature_dims (List[int]): the dimension of features in each stage.
    """

    def __init__(self):
        super(BaseConvModel, self).__init__()
        self.depth = cfg.CNN.DEPTH
        self.img_size = cfg.MODEL.IMG_SIZE
        self.in_channels = cfg.MODEL.IN_CHANNELS
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.features = list()
        self.feature_dims = None

    def initialize_hooks(self, layers, feature_dims):
        """
        Initialize hooks for the given layers.
        """
        for layer in layers:
            layer.register_forward_hook(self._feature_hook)
        self.feature_dims = feature_dims
        self.register_forward_pre_hook(lambda module, inp: self.features.clear())

    @abstractmethod
    def _feature_hook(self, module, inputs, outputs):
        pass

    def complexity(self):
        params = sum(p.numel() for p in self.parameters())
        return {'params': f'{round(params/1e6, 2)}M'}
