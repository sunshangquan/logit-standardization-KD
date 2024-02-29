import torch.nn as nn
from abc import ABCMeta, abstractmethod

from pycls.core.config import cfg


class BaseTransformerModel(nn.Module, metaclass=ABCMeta):
    """
    Base class for Transformer models.

    Attributes:
        - self.features (List[Tensor]): the features in each block.
        - self.feature_dims (List[int]): the dimension of features in each block.
        - self.distill_logits (Tensor|None): the logits of the distillation token, only for DeiT.
    """

    def __init__(self):
        super(BaseTransformerModel, self).__init__()
        # Base configs for Transformers
        self.img_size = cfg.MODEL.IMG_SIZE
        self.patch_size = cfg.TRANSFORMER.PATCH_SIZE
        self.patch_stride = cfg.TRANSFORMER.PATCH_STRIDE
        self.patch_padding = cfg.TRANSFORMER.PATCH_PADDING
        self.in_channels = cfg.MODEL.IN_CHANNELS
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.hidden_dim = cfg.TRANSFORMER.HIDDEN_DIM
        self.depth = cfg.TRANSFORMER.DEPTH
        self.num_heads = cfg.TRANSFORMER.NUM_HEADS
        self.mlp_ratio = cfg.TRANSFORMER.MLP_RATIO
        self.drop_rate = cfg.TRANSFORMER.DROP_RATE
        self.drop_path_rate = cfg.TRANSFORMER.DROP_PATH_RATE
        self.attn_drop_rate = cfg.TRANSFORMER.ATTENTION_DROP_RATE

        # Calculate the dimension of features in each block
        if isinstance(self.hidden_dim, int):
            assert isinstance(self.depth, int)
            self.feature_dims = [self.hidden_dim] * self.depth
        elif isinstance(self.hidden_dim, (list, tuple)):
            assert isinstance(self.depth, (list, tuple))
            assert len(self.hidden_dim) == len(self.depth)
            self.feature_dims = sum([[self.hidden_dim[i]] * d for i, d in enumerate(self.depth)], [])
        else:
            raise ValueError
        self.features = list()
        self.distill_logits = None

    def initialize_hooks(self, layers):
        """
        Initialize hooks for the given layers.
        """
        for layer in layers:
            layer.register_forward_hook(self._feature_hook)
        self.register_forward_pre_hook(lambda module, inp: self.features.clear())

    @abstractmethod
    def _feature_hook(self, module, inputs, outputs):
        pass

    def complexity(self):
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'params': f'{round(params/1e6, 2)}M'}
