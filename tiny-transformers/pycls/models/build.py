from fvcore.common.registry import Registry

from pycls.core.config import cfg
from .distill import DistillationWrapper


MODEL = Registry('MODEL')


def build_model():
    model = MODEL.get(cfg.MODEL.TYPE)()
    if cfg.DISTILLATION.ENABLE_INTER or cfg.DISTILLATION.ENABLE_LOGIT:
        teacher_mode = MODEL.get(cfg.DISTILLATION.TEACHER_MODEL)()
        model = DistillationWrapper(model, teacher_mode)
    return model
