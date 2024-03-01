from __future__ import division, print_function

import sys
import time

import torch
# from .distiller_zoo import DistillKL2
import torch.nn as nn
import torch.nn.functional as F

from .util import AverageMeter, accuracy, reduce_tensor


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) 

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        
        input, target = batch_data
        
        data_time.update(time.time() - end)
        
        # input = input.float()
        if opt.gpu is not None:
            input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        # output = model(input, is_feat=True)

        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, target, topk=(1, 5))
        top1.update(metrics[0].item(), input.size(0))
        top5.update(metrics[1].item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                   epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
            
    return top1.avg, top5.avg, losses.avg

def train_distill(epoch, train_loader, module_list, mlp_net, cos_value, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    
    if opt.have_mlp:
        mlp_net.train()

    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_kl = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target = data
            if opt.distill == 'semckd' and input.shape[0] < opt.batch_size:
                continue
        
        if opt.gpu is not None:
            input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        feat_s, logit_s = model_s(input, is_feat=True)

        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        if opt.have_mlp:
            temp = mlp_net(logit_t, logit_s, cos_value)  # (teacher_output, student_output)
            temp = opt.t_start + opt.t_end * torch.sigmoid(temp)
            temp = temp.cuda()
        else:
            temp = (opt.kd_T * torch.ones(1)).cuda()

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t, temp)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0                                    
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'srrl':
            cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else model_t.get_feat_modules()[-1]
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
            loss_kd = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, logit_t)
        elif opt.distill == 'dkd':
            loss_kd = criterion_kd(logit_s, logit_t, target, opt.dkd_alpha, opt.dkd_beta, temp)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        
        loss_kl.update(loss_div.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        metrics = accuracy(logit_s, target, topk=(1, 5))
        top1.update(metrics[0].item(), input.size(0))
        top5.update(metrics[1].item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, loss=losses,
                top1=top1, top5=top5
                ))
            sys.stdout.flush()
            # print(temp)

    return top1.avg, top5.avg, losses.avg, temp


def validate(val_loader, model, criterion, opt):
    """validation"""
    
    # batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    n_batch = len(val_loader)

    with torch.no_grad():
        # end = time.time()
        for idx, batch_data in enumerate(val_loader):
            
            input, target = batch_data

            if opt.gpu is not None:
                input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure accuracy and record loss
            metrics = accuracy(output, target, topk=(1, 5))
            top1.update(metrics[0].item(), input.size(0))
            top5.update(metrics[1].item(), input.size(0))

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                        'GPU: {2}\t'
                        'Loss {loss.avg:.4f}\t'
                        'Acc@1 {top1.avg:.3f}\t'
                        'Acc@5 {top5.avg:.3f}'.format(
                        idx, n_batch, opt.gpu, loss=losses,
                        top1=top1, top5=top5))
    
    return [top1.avg, top5.avg, losses.avg]
