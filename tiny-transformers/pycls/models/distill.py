import torch
import torch.nn as nn
import torch.nn.functional as F

from pycls.core.config import cfg
import pycls.core.logging as logging


logger = logging.get_logger(__name__)


def attention_transform(feat):
    return F.normalize(feat.pow(2).mean(1).view(feat.size(0), -1))


def similarity_transform(feat):
    feat = feat.view(feat.size(0), -1)
    gram = feat @ feat.t()
    return F.normalize(gram)


_TRANS_FUNC = {"attention": attention_transform, "similarity": similarity_transform, "linear": lambda x : x}


def inter_distill_loss(feat_t, feat_s, transform_type):
    assert transform_type in _TRANS_FUNC, f"Transformation function {transform_type} is not supported."
    trans_func = _TRANS_FUNC[transform_type]
    feat_t = trans_func(feat_t)
    feat_s = trans_func(feat_s)
    return (feat_t - feat_s).pow(2).mean()


def logit_distill_loss(logits_t, logits_s, loss_type, temperature):
    if loss_type == "soft":
        distillation_loss = F.kl_div(
            F.log_softmax(logits_s / temperature, dim=1),
            F.log_softmax(logits_t / temperature, dim=1),
            reduction='sum',
            log_target=True
        ) * (temperature * temperature) / logits_s.numel()
    elif loss_type == "hard":
        distillation_loss = F.cross_entropy(logits_s, logits_t.argmax(dim=1))
    else:
        raise NotImplementedError

    return distillation_loss


class DistillationWrapper(nn.Module):

    def __init__(self, student_model, teacher_mode):
        super(DistillationWrapper, self).__init__()
        self.enable_inter = cfg.DISTILLATION.ENABLE_INTER
        self.inter_transform_type = cfg.DISTILLATION.INTER_TRANSFORM
        self.student_idx = cfg.DISTILLATION.INTER_STUDENT_INDEX
        self.teacher_idx = cfg.DISTILLATION.INTER_TEACHER_INDEX
        self.enable_logit = cfg.DISTILLATION.ENABLE_LOGIT
        self.logit_loss_type = cfg.DISTILLATION.LOGIT_LOSS
        self.teacher_img_size = cfg.DISTILLATION.TEACHER_IMG_SIZE
        self.offline = cfg.DISTILLATION.OFFLINE
        assert not self.offline or not self.enable_logit, 'Logit distillation is not supported when offline is enabled.'

        self.student_model = student_model

        self.teacher_model = teacher_mode
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        logger.info("Build teacher model {}".format(type(self.teacher_model)))

        teacher_weights = cfg.DISTILLATION.TEACHER_WEIGHTS
        if teacher_weights:
            checkpoint = torch.load(teacher_weights)["model_state"]
            logger.info("Loaded initial weights of teacher model from: {}".format(teacher_weights))
            self.teacher_model.load_state_dict(checkpoint)

        if self.inter_transform_type == 'linear':
            self.feature_transforms = nn.ModuleList()
            for i, j in zip(self.student_idx, self.teacher_idx):
                self.feature_transforms.append(nn.Conv2d(self.student_model.feature_dims[i], self.teacher_model.feature_dims[j], 1))

    def load_state_dict(self, state_dict, strict=True):
        return self.student_model.load_state_dict(state_dict, strict)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.student_model.state_dict(destination, prefix, keep_vars)

    def forward(self, x):
        return self.student_model(x)

    def complexity(self):
        complexity = dict()
        student_complexity = self.student_model.complexity()
        teacher_complexity = self.teacher_model.complexity()
        complexity["student"] = student_complexity
        complexity["teacher"] = teacher_complexity
        return complexity

    def guidance_loss(self, x, offline_feats):
        logits_s = self.student_model.distill_logits
        feats_s = self.student_model.features

        if self.offline:
            logits_t = None
            feats_t = offline_feats
        else:
            x = F.interpolate(x, size=(self.teacher_img_size, self.teacher_img_size), mode='bilinear', align_corners=False)
            with torch.no_grad():
                logits_t = self.teacher_model(x)
                feats_t = self.teacher_model.features

        loss_inter = x.new_tensor(0.0)
        if self.enable_inter:
            for i, (idx_t, idx_s) in enumerate(zip(self.teacher_idx, self.student_idx)):
                feat_t = feats_t[idx_t]
                feat_s = feats_s[idx_s]

                if self.inter_transform_type == 'linear':
                    feat_s = self.feature_transforms[i](feat_s)

                dsize = (max(feat_t.size(-2), feat_s.size(-2)), max(feat_t.size(-1), feat_s.size(-1)))
                feat_t = F.interpolate(feat_t, dsize, mode='bilinear', align_corners=False)
                feat_s = F.interpolate(feat_s, dsize, mode='bilinear', align_corners=False)
                loss_inter = loss_inter + inter_distill_loss(feat_t, feat_s, self.inter_transform_type)

        loss_logit = logit_distill_loss(logits_t, logits_s, self.logit_loss_type) if self.enable_logit else x.new_tensor(0.0)

        return loss_inter, loss_logit
