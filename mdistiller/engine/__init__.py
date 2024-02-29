from .trainer import BaseTrainer, CRDTrainer, AugTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "ours": AugTrainer
}
