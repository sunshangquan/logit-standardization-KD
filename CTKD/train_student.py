"""
the general training framework
"""

from __future__ import print_function

import argparse
import json
# import tensorboard_logger as tb_logger
import logging
import math
import os
import re
import time

import numpy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from crd.criterion import CRDLoss
from dataset.cifar100 import (get_cifar100_dataloaders,
                              get_cifar100_dataloaders_sample)
from dataset.imagenet import get_imagenet_dataloader, imagenet_list
from distiller_zoo import PKT, DistillKL, DKDloss, Similarity, VIDLoss, DistillKL_logit_stand
from helper.loops import train_distill as train
from helper.loops import validate
from helper.util import (adjust_learning_rate, parser_config_save,
                         reduce_tensor, save_dict_to_json)
from models import model_dict

from models.temp_global import Global_T
from models.util import SRRL, ConvReg, Embed, LinearEmbed, SelfA

split_symbol = '~' if os.name == 'nt' else ':'


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    # basic
    parser.add_argument('--print-freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--experiments_dir', type=str, default='models',help='Directory name to save the model, log, config')
    parser.add_argument('--experiments_name', type=str, default='baseline')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'imagenette'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'ResNet18', 'ResNet34', 'resnet8x4_double',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'wrn_50_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg11_imagenet', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_Imagenet', 'MobileNetV2_Imagenet',
                                 'shufflenet_v2_x0_5', 'shufflenet_v2_x2_0', 'ResNet18Double'])
    parser.add_argument('--path-t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'similarity', 'vid',
                                                                       'pkt', 'crd', 'dkd', 'srrl'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=0.1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0.9, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='default temperature for KD distillation')

    # CTKD distillation
    parser.add_argument('--have_mlp', type=int, default=0)
    parser.add_argument('--mlp_name', type=str, default='global')
    parser.add_argument('--t_start', type=float, default=1)
    parser.add_argument('--t_end', type=float, default=20)
    parser.add_argument('--cosine_decay', type=int, default=1)
    parser.add_argument('--decay_max', type=float, default=0)
    parser.add_argument('--decay_min', type=float, default=0)
    parser.add_argument('--decay_loops', type=float, default=0)

    # DKD distillation
    parser.add_argument('--dkd_alpha', default=1, type=float)
    parser.add_argument('--dkd_beta', default=2, type=float)

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--logit_stand', action='store_true')
    # switch for edge transformation
    parser.add_argument('--no_edge_transform', action='store_true') # default=false
    
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:8080', type=str,
                    help='url used to set up distributed training')
    
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard
    opt.model_path = './save/student_model'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name =  os.path.join(opt.experiments_dir, opt.experiments_name)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    parser_config_save(opt, opt.save_folder)

    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    if segments[0] == 'resnext50':
        return segments[0] + '_' + segments[1]
    if segments[0] == 'vgg13' and segments[1] == 'imagenet':
        return segments[0] + '_' + segments[1]
    return segments[0]


def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    # TODO: reduce size of the teacher saved in train_teacher.py
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}

    if opt.dataset == 'cifar100':
        model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    elif opt.dataset == 'imagenet':
        checkpoint = torch.load(model_path, map_location=map_location)
        # new_state_dict = {}
        # for k,v in checkpoint['model'].items():
        #     new_state_dict[k[7:]] = v
        # model.load_state_dict(checkpoint['state'])
        model.load_state_dict(checkpoint)
    
    print('==> done')
    return model


class CosineDecay(object):
    def __init__(self,
                max_value,
                min_value,
                num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class LinearDecay(object):
    def __init__(self,
                max_value,
                min_value,
                num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops - 1

        value = (self._max_value - self._min_value) / self._num_loops
        value = i * (-value)

        return value


total_time = time.time()
best_acc = 0

def main():
    
    opt = parse_option()
    
    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    
    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if opt.deterministic:
        torch.manual_seed(27)
        cudnn.deterministic = False
        cudnn.benchmark = True
        numpy.random.seed(27)

    class_num_map = {
        'cifar100': 100,
        'imagenet': 1000,
        'imagenette': 10,
    }
    if opt.dataset not in class_num_map:
        raise NotImplementedError(opt.dataset)
    n_cls = class_num_map[opt.dataset]

    # model
    model_t = load_teacher(opt.path_t, n_cls, opt.gpu, opt)
    
    module_args = {'num_classes': n_cls}
    model_s = model_dict[opt.model_s](**module_args)
    
    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'imagenet':
        data = torch.randn(2, 3, 224, 224)

    mlp = None

    if opt.have_mlp:
        if opt.mlp_name == 'global':
            mlp = Global_T()
        else:
            print('mlp name wrong')

    model_t.eval()
    model_s.eval()

    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)
    
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    trainable_list.append(mlp)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL() if opt.logit_stand else DistillKL_logit_stand()
    if opt.distill == 'kd':
        criterion_kd = DistillKL()
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = 50000
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'dkd':
        criterion_kd = DKDloss()
    elif opt.distill == 'srrl':
        s_n = feat_s[-1].shape[1]
        t_n = feat_t[-1].shape[1]
        model_fmsr = SRRL(s_n= s_n, t_n=t_n)
        criterion_kd = nn.MSELoss()
        module_list.append(model_fmsr)
        trainable_list.append(model_fmsr)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    module_list.append(model_t)
    
    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                module_list.cuda(opt.gpu)
                distributed_modules = []
                for module in module_list:
                    DDP = torch.nn.parallel.DistributedDataParallel
                    distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
                module_list = distributed_modules
                criterion_list.cuda(opt.gpu)
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion_list.cuda()
            module_list.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers)
    elif opt.dataset in imagenet_list:
        train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset=opt.dataset, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        multiprocessing_distributed=opt.multiprocessing_distributed)
    else:
        raise NotImplementedError(opt.dataset)

    if opt.cosine_decay:
        gradient_decay = CosineDecay(max_value=opt.decay_max, min_value=opt.decay_min, num_loops=opt.decay_loops)
    else:
        gradient_decay = LinearDecay(max_value=opt.decay_max, min_value=opt.decay_min, num_loops=opt.decay_loops)

    decay_value = 1
    best_acc = 0

    for epoch in range(1, opt.epochs + 1):

        torch.cuda.empty_cache()
        if opt.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        if opt.have_mlp:
            decay_value = gradient_decay.get_value(epoch)

        train_acc, train_acc_top5, train_loss, temp = train(epoch, train_loader, module_list, mlp, decay_value, criterion_list, optimizer, opt)

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()


        return_pack = validate(val_loader, model_s, criterion_cls, opt)        
        test_acc, test_acc_top5, _ = return_pack

        best_model = False
        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = True

            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            
            test_merics = {
                            'test_acc': test_acc,
                            'test_acc_top5': test_acc_top5,
                            'best_acc': best_acc,
                            'epoch': epoch,
                            'temp': json.dumps(temp.cpu().detach().numpy()[0].tolist()),
                            'decay_value': decay_value}

            save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))

            if epoch > opt.epochs/2:
                if best_model:
                    best_model=False
                    if opt.save_model:
                        torch.save(state, save_file)


if __name__ == '__main__':
    main()
