# adapted from https://github.com/microsoft/mup/blob/main/examples/ResNet/resnet.py

# only differs in terms of initialization in mup, which only differs in last layer from sp and ntp.
# changes concerning training are done in other files.
# conv: conv1 input, else hidden-like, is correctly initialized with kaiming
# batchnorm: input-like, is already correctly initialized: weights=1, biases=0

'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''


import fnmatch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utils import get_bcd

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, in_multiplier = 1, hidden_multiplier = 1):
        super(BasicBlock, self).__init__()
        self.hidden_multiplier = hidden_multiplier
        self.in_multiplier = in_multiplier
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.long_shortcut = True
            self.conv_shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,stride=stride, bias=False)
            self.bn_shortcut = nn.BatchNorm2d(self.expansion*planes)
        else:
            self.long_shortcut=False
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        layers = [self.conv1, self.conv2]
        if self.long_shortcut:
            layers.append(self.conv_shortcut)
        for layer in layers:
            init.kaiming_normal_(layer.weight, a=1)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, x, out_sublayer=None):
        out = self.conv1(self.hidden_multiplier*x)
        if out_sublayer == 1:
            return out 
        out = F.relu(self.bn1(self.in_multiplier*out))
        if out_sublayer == 2:
            return out
        out = self.conv2(self.hidden_multiplier*out)
        if out_sublayer == 3:
            return out
        out = self.bn2(self.in_multiplier*out)
        if out_sublayer == 4:
            return out
        if self.long_shortcut:
            out += self.bn_shortcut(self.in_multiplier * self.conv_shortcut(self.hidden_multiplier * x))
        else:
            out += x
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
            self.reset_parameters()

    def reset_parameters(self) -> None:
        layers = [self.conv1, self.conv2, self.conv3]
        if len(self.shortcut) > 1:
            layers.append(self.shortcut[0])
        for layer in layers:
            init.kaiming_normal_(layer.weight, a=1)
            if layer.bias is not None:
                init.zeros_(layer.bias)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, parameterization='mup', perturbation='mpp', variant='sam',
                 num_classes=10, wm=1, init_var=2, ll_zero_init=True,
                 out_multiplier = 4, in_multiplier = 1, hidden_multiplier = 1,
                 base_wm = None,split_gpus: int = None): # out_multiplier = 4 optimal from TP5 and our search
        super(ResNet, self).__init__()

        base_widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in base_widths]
        self.widths = widths
        
        self.parameterization = parameterization
        self.perturbation = perturbation
        self.variant = variant
        self.ll_zero_init = ll_zero_init
        self.out_multiplier = out_multiplier
        self.in_multiplier = in_multiplier
        self.hidden_multiplier = hidden_multiplier
        self.base_wm = base_wm
        self.split_gpus = split_gpus

        # scalings for input-, hidden- and output-like params
        bl, cl, dl, d = get_bcd(L=2, param=parameterization, perturb=perturbation,variant=self.variant)

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)

        self.outlayer = nn.Linear(widths[3] * block.expansion, num_classes)
        
        if self.split_gpus is not None:
            if self.split_gpus == 2:
                gpu_per_layer = [0,0,1,1,1]
            elif self.split_gpus == 3:
                gpu_per_layer = [0,1,1,2,2]
            elif self.split_gpus == 4:
                gpu_per_layer = [0,1,2,3,3]
            elif self.split_gpus >= 5:
                gpu_per_layer = [0,1,2,3,4]
            else:
                raise ValueError('split_gpus should be an int >= 2.')
                
            self.conv1.cuda(gpu_per_layer[0])
            self.bn1.cuda(gpu_per_layer[0])
            self.layer1.cuda(gpu_per_layer[0])
            self.layer2.cuda(gpu_per_layer[1])
            self.layer3.cuda(gpu_per_layer[2])
            self.layer4.cuda(gpu_per_layer[3])
            self.outlayer.cuda(gpu_per_layer[4])
            
        
        if ll_zero_init:
            self.outlayer.weight.data[:] = 0
        else:
            if self.parameterization == 'mup':
                if base_wm is None:
                    #kaiming with scaling
                    self.outlayer.weight.data.normal_(mean=0,std=float(np.sqrt(num_classes * init_var)/ (widths[3] * block.expansion)))
                else:
                    ll_basewidth = int(base_widths[-1]*base_wm)*block.expansion
                    #kaiming with scaling, equivalent to sp at base_wm
                    self.outlayer.weight.data.normal_(mean=0,std=float(np.sqrt(num_classes * init_var)/ (widths[3] * block.expansion) * np.sqrt(ll_basewidth)/np.sqrt(num_classes)))
            else:
                self.outlayer.weight.data.normal_(mean=0,std=float(np.sqrt(init_var/ (widths[3] * block.expansion))))
        if self.outlayer.bias is not None:
            self.outlayer.bias.data.zero_()
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride,in_multiplier = self.in_multiplier, hidden_multiplier = self.hidden_multiplier))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_layer = None):
        if self.split_gpus is not None: x = x.cuda(0)
        out = F.relu(self.bn1(self.in_multiplier * self.conv1(self.in_multiplier * x)))
        out = self.layer1(out)
        if out_layer == 1: return out
        if self.split_gpus is not None and self.split_gpus >= 3:
            out = out.cuda(1)
        out = self.layer2(out)
        if out_layer == 2: return out
        if self.split_gpus is not None:
            if self.split_gpus >= 4:
                out = out.cuda(2)
            elif self.split_gpus == 2:
                out = out.cuda(1)
        out = self.layer3(out)
        if out_layer == 3: return out
        if self.split_gpus is not None:
            if self.split_gpus >= 4:
                out = out.cuda(3)
            elif self.split_gpus == 3:
                out = out.cuda(2)
        out = self.layer4(out)
        if out_layer == 4: return out
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        if self.split_gpus is not None and self.split_gpus >= 5:
            pre_out = pre_out.cuda(4)
        final = self.out_multiplier * self.outlayer(pre_out)
        return final
    
    
    def get_bcd(self,param = None,perturb=None,variant=None):
        """For each trainable parameter, returns initialization, learning rate and perturbation exponent corresponding to the pre-specified parameterization."""
        # first layer, bias and bn always input,
        # last layer always output
        # conv hidden
        # shortcut.0=conv
        # shortcut.1=bn
        if param is None:
            param = self.parameterization
        if perturb is None:
            perturb = self.perturbation
        if variant is None:
            variant = self.variant

        # scalings for input-, hidden- and output-like params
        bl, cl, dl, d = get_bcd(L=2, param=param, perturb=perturb,variant=variant)
        bls, cls, dls = [],[],[]
        for name, param in self.named_parameters():
            if (fnmatch.fnmatch(name,'conv1.*') or fnmatch.fnmatch(name,'*bn*.weight')
                or fnmatch.fnmatch(name,'*.bias') or fnmatch.fnmatch(name,'*.shortcut.1*')):
                bls.append(bl[0])
                if fnmatch.fnmatch(name,'outlayer.bias'):
                    cls.append(0)
                    dls.append(0)
                else:
                    cls.append(cl[0])
                    dls.append(dl[0])
            elif fnmatch.fnmatch(name,'*conv*.weight') or fnmatch.fnmatch(name,'*.shortcut.0*'):
                bls.append(bl[1])
                cls.append(cl[1])
                dls.append(dl[1])
            elif fnmatch.fnmatch(name, 'outlayer.weight'):
                bls.append(bl[-1])
                cls.append(cl[-1])
                dls.append(dl[-1])
            else:
                raise NameError(name+' is not a known layer.')
        return bls, cls, dls, d
    
    def _disable_running_stats(self, m):
        if isinstance(m, nn.BatchNorm2d):
            m.backup_momentum = m.momentum
            m.momentum = 0
            
    def _enable_running_stats(self, m):
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = m.backup_momentum
                

def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def ResNet18Wide(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wm=5, **kwargs)

def ResNet18Thin(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wm=.75, **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)


# from https://github.com/davda54/sam/issues/30
# for SAM model.apply(self._disable_running_stats)
# and model.apply(self._enable_running_stats)
# see sam.py and resnet_trainer.py

# toy_resnet = ResNet18(wm=0.5)
# for name, param in toy_resnet.named_parameters():
#     print(name, param.size())
