from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvReg(nn.Module):
    """Convolutional regression for FitNet (feature map layer)"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        self.s_H = s_H
        self.t_H = t_H
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        if self.s_H == 2 * self.t_H or self.s_H * 2 == self.t_H or self.s_H >= self.t_H:
            x = self.conv(x)
            if self.use_relu:
                return self.relu(self.bn(x)), t
            else:
                return self.bn(x), t
        else:
            x = self.conv(x)
            if self.use_relu:
                return self.relu(self.bn(x)), F.adaptive_avg_pool2d(t, (self.s_H, self.s_H))
            else:
                return self.bn(x), F.adaptive_avg_pool2d(t, (self.s_H, self.s_H))

class Regress(nn.Module):
    """Simple Linear Regression for FitNet (feature vector layer)"""
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Regress, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)
        return x

class SelfA(nn.Module):
    """Cross layer Self Attention"""
    def __init__(self, s_len, t_len, input_channel, s_n, s_t, factor=4): 
        super(SelfA, self).__init__()
          
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        for i in range(t_len):
            setattr(self, 'key_weight'+str(i), MLPEmbed(input_channel, input_channel//factor))
        for i in range(s_len):
            setattr(self, 'query_weight'+str(i), MLPEmbed(input_channel, input_channel//factor))
        
        for i in range(s_len):
            for j in range(t_len):
                setattr(self, 'regressor'+str(i)+str(j), AAEmbed(s_n[i], s_t[j]))
               
    def forward(self, feat_s, feat_t):
        
        sim_t = list(range(len(feat_t)))
        sim_s = list(range(len(feat_s)))
        bsz = feat_s[0].shape[0]
        # similarity matrix
        for i in range(len(feat_t)):
            sim_temp = feat_t[i].reshape(bsz, -1)
            sim_t[i] = torch.matmul(sim_temp, sim_temp.t())
        for i in range(len(feat_s)):
            sim_temp = feat_s[i].reshape(bsz, -1)
            sim_s[i] = torch.matmul(sim_temp, sim_temp.t())
        
        # key of target layers    
        proj_key = self.key_weight0(sim_t[0])
        proj_key = proj_key[:, :, None]
        
        for i in range(1, len(sim_t)):
            temp_proj_key = getattr(self, 'key_weight'+str(i))(sim_t[i])
            proj_key =  torch.cat([proj_key, temp_proj_key[:, :, None]], 2)
        
        # query of source layers   
        proj_query = self.query_weight0(sim_s[0])
        proj_query = proj_query[:, None, :]
        for i in range(1, len(sim_s)):
            temp_proj_query = getattr(self, 'query_weight'+str(i))(sim_s[i])
            proj_query = torch.cat([proj_query, temp_proj_query[:, None, :]], 1)
        
        # attention weight
        energy = torch.bmm(proj_query, proj_key) # batch_size X No.stu feature X No.tea feature
        attention = F.softmax(energy, dim = -1)
        
        # feature space alignment
        proj_value_stu = []
        value_tea = []
        for i in range(len(sim_s)):
            proj_value_stu.append([])
            value_tea.append([])
            for j in range(len(sim_t)):            
                s_H, t_H = feat_s[i].shape[2], feat_t[j].shape[2]
                if s_H > t_H:
                    input = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
                    proj_value_stu[i].append(getattr(self, 'regressor'+str(i)+str(j))(input))
                    value_tea[i].append(feat_t[j])
                elif s_H < t_H or s_H == t_H:
                    target = F.adaptive_avg_pool2d(feat_t[j], (s_H, s_H))
                    proj_value_stu[i].append(getattr(self, 'regressor'+str(i)+str(j))(feat_s[i]))
                    value_tea[i].append(target)
                
        return proj_value_stu, value_tea, attention
           
class AAEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(AAEmbed, self).__init__()
        self.num_mid_channel = 2 * num_target_channels
        
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x
        
class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class LinearEmbed(nn.Module):
    """Linear Embedding"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """flatten module"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class PoolEmbed(nn.Module):
    """pool and embed"""
    def __init__(self, layer=0, dim_out=128, pool_type='avg'):
        super().__init__()
        if layer == 0:
            pool_size = 8
            nChannels = 16
        elif layer == 1:
            pool_size = 8
            nChannels = 16
        elif layer == 2:
            pool_size = 6
            nChannels = 32
        elif layer == 3:
            pool_size = 4
            nChannels = 64
        elif layer == 4:
            pool_size = 1
            nChannels = 64
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.embed = nn.Sequential()
        if layer <= 3:
            if pool_type == 'max':
                self.embed.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.embed.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.embed.add_module('Flatten', Flatten())
        self.embed.add_module('Linear', nn.Linear(nChannels*pool_size*pool_size, dim_out))
        self.embed.add_module('Normalize', Normalize(2))

    def forward(self, x):
        return self.embed(x)


class SRRL(nn.Module):
    """ICLR-2021: Knowledge Distillation via Softmax Regression Representation Learning"""
    def __init__(self, *, s_n, t_n): 
        super(SRRL, self).__init__()
                
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        
        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        ))
        
    def forward(self, feat_s, cls_t):
        feat_s = feat_s.unsqueeze(-1).unsqueeze(-1)
        temp_feat = self.transfer(feat_s)
        trans_feat_s = temp_feat.view(temp_feat.size(0), -1)
        pred_feat_s=cls_t(trans_feat_s)
        return trans_feat_s, pred_feat_s

class MGD(nn.Module):
    def __init__(self, student_channels, teacher_channels, mgd_lambda):
        super(MGD, self).__init__()
        
        # self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1)
        )

        self.mgd_lambda = mgd_lambda

    def forward(self, x):

        device = x.device
        N, C, H, W = x.shape
        mat = torch.rand((N, C, H, W))
        mat = torch.where(mat < self.mgd_lambda, 0, 1).to(device)

        masked_fea = torch.mul(x, mat)
        masked_fea = self.generation(masked_fea)

        return masked_fea


class custom_MGD(nn.Module):
    def __init__(self):
        super(custom_MGD, self).__init__()
     
        channels=512
        self.generation = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        masked_fea = self.generation(x)

        return masked_fea


class Embed(nn.Module):
    def __init__(self, dim_in=256, dim_out=128):
        super(Embed, self).__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out)
        )
    def forward(self, x):
        x = self.operation(x)
        return x


if __name__ == '__main__':
    import torch
    # from torchmetrics.functional import pairwise_euclidean_distance

    a = torch.rand([32, 512, 512])
    b = torch.rand([32, 512, 512])
    c = nn.PairwiseDistance(p=2)(a,b)
    print(c.shape)
    # b = torch.rand([32, 512])
    
    # metric = torch.nn.PairwiseDistance(p=2)
    # result = metric(a,b).mean(2).mean(1).mean(0)

    # result = (a - b).pow(2).sum(3).sqrt()
    # print(result)
    # print(pairwise_euclidean_distance(a, b,reduction='mean'))

    # print(a.transpose(2,1).shape)