#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import math
from torch.nn import Parameter
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F


class ReLU(nn.Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
               + inplace_str + ')'


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Instance-Batch Normalization Block
# https://github.com/XingangPan/IBN-Net
class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out




# 标准resnet block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# IBN_a_Bottleneck
# https://github.com/XingangPan/IBN-Net
class IBN_a_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(IBN_a_Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = IBN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Squeeze-and-Excitation block
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.globalAvgPool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc1 = nn.Linear(in_features=planes, out_features=int(round(planes / 16)))
        self.fc2 = nn.Linear(in_features=int(round(planes / 16)), out_features=planes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        out += residual
        out = self.relu(out)
        return out


# 网络结构
class myResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, ratio=1.0):
        super(myResNet, self).__init__()

        self.relu = ReLU(inplace=True)
        self.inplanes = int(32 * ratio)
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(block, self.inplanes, layers[0])

        self.inplanes = int(64 * ratio)
        self.conv2 = nn.Conv2d(int(32 * ratio), self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(self.inplanes)
        self.layer2 = self._make_layer(block, self.inplanes, layers[1])

        self.inplanes = int(128 * ratio)
        self.conv3 = nn.Conv2d(int(64 * ratio), self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(self.inplanes)
        self.layer3 = self._make_layer(block, self.inplanes, layers[2])

        self.inplanes = int(256 * ratio)
        self.conv4 = nn.Conv2d(int(128 * ratio), self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(self.inplanes)
        self.layer4 = self._make_layer(block, self.inplanes, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d([4, 1])
        self.fc = nn.Linear(self.inplanes * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# consieLinear层 实现了norm的fea与norm weight的点积计算，服务于margin based softmax loss
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)  # weights normed
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)  x.dot(ww)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)  #
        cos_theta = cos_theta.clamp(-1, 1)
        cos_theta = cos_theta * xlen.view(-1, 1)

        return cos_theta  # size=(B,Classnum,1)


# AMSoftmax 层的pytorch实现，两个重要参数 scale，margin（不同难度和量级的数据对应不同的最优参数）
# 原始实现caffe->https://github.com/happynear/AMSoftmax
class AMSoftmax(nn.Module):
    def __init__(self):
        super(AMSoftmax, self).__init__()

    def forward(self, input, target, scale=10.0, margin=0.35):
        # self.it += 1
        cos_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= margin
        output = output * scale

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss


# 网络结构
class DeepSpeakerModel(nn.Module):
    def __init__(self, embedding_size, num_classes, ratio=1.0, is_fn=False, is_dropout=False, is_SEnet=False):
        super(DeepSpeakerModel, self).__init__()

        self.is_fn = is_fn
        self.is_dropout = is_dropout

        self.embedding_size = embedding_size

        if is_SEnet:
            self.model = myResNet(SEBasicBlock, [1, 1, 1, 1], ratio=ratio)
        else:
            self.model = myResNet(BasicBlock, [1, 1, 1, 1], ratio=ratio)

        # self.model = myResNet(IBN_a_Bottleneck, [1, 1, 1, 1], ratio = ratio)

        # feature norm trick，无bias的bn层加在特征层之后，可以提升verification特征性能
        if self.is_fn:
            self.model.fc = nn.Linear(int(256 * ratio * 4), self.embedding_size, bias=False)
            self.fn = nn.BatchNorm2d(embedding_size, affine=False)
        else:
            self.model.fc = nn.Linear(int(256 * ratio * 4), self.embedding_size)

        if self.is_dropout:
            self.dp = nn.Dropout(0.5)

        self.model.classifier = CosineLinear(self.embedding_size, num_classes)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.is_dropout:
            x = self.dp(x)

        x = self.model.fc(x)

        if self.is_fn:
            x = self.fn(x)

        self.features = self.l2_norm(x)

        x = self.model.classifier(self.features)
        return self.features, x

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res


# class PairwiseDistance(Function):
#     def __init__(self, p):
#         super(PairwiseDistance, self).__init__()
#         self.norm = p
#
#     def forward(self, x1, x2):
#         assert x1.size() == x2.size()
#         eps = 1e-4 / x1.size(1)
#         diff = torch.abs(x1 - x2)
#         out = torch.pow(diff, self.norm).sum(dim=1)
#         return torch.pow(out + eps, 1. / self.norm)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res





