from __future__ import print_function

import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.imagenet import get_imagenet_dataloader, imagenet_list
from helper.loops import train_vanilla as train
from helper.loops import validate
from helper.util import (AverageMeter, accuracy, adjust_learning_rate,
                         parser_config_save, reduce_tensor, save_dict_to_json)
from models import model_dict


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # baisc
    parser.add_argument('--print-freq', type=int, default=200, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0,1,2,3,4,5,6,7', help='id(s) for CUDA_VISIBLE_DEVICES')
    
    parser.add_argument('--experiments_dir', type=str, default='models',help='Directory name to save the model, log, config')
    parser.add_argument('--experiments_name', type=str, default='baseline')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'ResNet18', 'ResNet34', 
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2','ResNet50' ])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'imagenette'], help='dataset')
    
    parser.add_argument('--use-lmdb', action='store_true') # default=false

    # multiprocessing
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
    
    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard 
    opt.model_path = './save/models'
    # opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name =  os.path.join(opt.experiments_dir, opt.experiments_name)

    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    parser_config_save(opt, opt.save_folder)

    return opt

best_acc = 0
total_time = time.time()
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
        opt.rank = int(gpu)
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

    # model
    n_cls = {
        'cifar100': 100,
        'imagenet': 1000,
        'imagenette': 10,
    }.get(opt.dataset, None)
    
    model = model_dict[opt.model](num_classes=n_cls, feat_drop_ratio=opt.feat_drop_ratio)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss().cuda()

    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                model.cuda(opt.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                opt.batch_size = int(opt.batch_size / ngpus_per_node)
                opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
                # DDP = torch.nn.parallel.DistributedDataParallel if opt.dali is None else apex.parallel.DistributedDataParallel
                # model = DDP(model, delay_allreduce=True)
                DDP = torch.nn.parallel.DistributedDataParallel
                model = DDP(model, device_ids=[opt.gpu])
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion = criterion.cuda()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=opt.gpu_id).cuda()
                # model = nn.DataParallel(model).cuda()
            else:
                model = model.cuda()

    cudnn.benchmark = True

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.dataset in imagenet_list:
        train_loader, val_loader, train_sampler = get_imagenet_dataloader(
                    dataset = opt.dataset,
                    batch_size=opt.batch_size, num_workers=opt.num_workers,
                    multiprocessing_distributed=opt.multiprocessing_distributed)
    else:
        raise NotImplementedError(opt.dataset)

    # tensorboard
    # if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
    #     logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):
        if opt.multiprocessing_distributed:
            if opt.dali is None:
                train_sampler.set_epoch(epoch)
            # No test_sampler because epoch is random seed, not needed in sequential testing.

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        train_acc, train_acc_top5, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, Acc@1 {:.3f}, Acc@5 {:.3f}'.format(epoch, train_acc, train_acc_top5))

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))
            
            test_merics = { 
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'test_acc_top5': test_acc_top5,
                    'epoch': epoch}    
            save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
            
if __name__ == '__main__':
    main()
