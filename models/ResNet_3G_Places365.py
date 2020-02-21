import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
__all__ = ['ResNet_3G_365', 'resnet18_3g_365', 'resnet34_3g_365', 'resnet23_3g_365', 'resnet50_3g_365', 'resnet101_3g_365',
           'resnet152_3g_365']



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def cov_forward(x):
    batchsize = x.data.shape[0]
    dim = x.data.shape[1]
    h = x.data.shape[2]
    w = x.data.shape[3]
    M = h*w
    x = x.reshape(batchsize,dim,M)
    I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
    I_hat = I_hat.view(1,M,M).repeat(batchsize,1,1).type(x.dtype)
    y = x.bmm(I_hat).bmm(x.transpose(1,2))

    I = torch.eye(dim, dim, device = x.device).view(1, dim, dim).repeat(batchsize,1,1)
    I_normX = I*1e-7

    y = y + I_normX
    return y


def sqrt_forward(x,numIters):
    batchSize = x.data.shape[0]
    dim = x.data.shape[1]
    dtype = x.dtype
    I = torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    normA = x.mul(I).sum(dim=1).sum(dim=1)
    Y = x.div(normA.view(batchSize, 1, 1).expand_as(x))
    Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,1,1).type(dtype)
    for i in range(numIters):
      T = 0.5*(3.0*I - Z.bmm(Y))
      Y = Y.bmm(T)
      Z = T.bmm(Z)
    y = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
    return y


def triu_forward(x):
    batchSize = x.data.shape[0]
    dim = x.data.shape[1]
    dtype = x.dtype
    x = x.reshape(batchSize, dim*dim)
    I = torch.ones(dim,dim).triu().t().reshape(dim*dim)
    id = I.nonzero()
    y = torch.zeros(batchSize,int(dim*(dim+1)/2),device = x.device)
    for i in range(batchSize):
        y[i, :] = x[i, id].t()
    return y

class AtMp(nn.Module):
  
    def __init__(self, n_features, reduction=4):
        super(AtMp, self).__init__()
        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction       (default = 16)')

        self.map1 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(n_features)

        self.map2 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(n_features)

        self.map21 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn21 = nn.BatchNorm2d(n_features)

        self.nonlin2 = nn.ReLU(inplace=True)


        self.map3 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn3 = nn.BatchNorm2d(n_features)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.map4 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn4 = nn.BatchNorm2d(n_features)
        self.nonlin4 = nn.ReLU(inplace=True)

        self.atmap1 = nn.Conv2d(n_features, int(n_features/reduction), kernel_size=1, stride=1, padding=0,bias=False)
        self.bnat1 = nn.BatchNorm2d(int(n_features/reduction))
        self.nonlinat1 = nn.ReLU(inplace=True)
        self.atmap2 = nn.Conv2d(int(n_features/reduction), 1, kernel_size=1, stride=1, padding=0,bias=False)

        self.atmap3 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0,bias=False)
       
        self.nonliner = nn.Sigmoid()
    
    def forward(self, x, z):    
        bs = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        a = x
        y = x
        x = self.map1(x)
        x = self.bn1(x)
        
        y = self.map2(y)
        y = self.bn2(y)
      
        x = x * y 
       
        z = self.map21(z)
        z = self.bn21(z)

        x = x + z
        x = self.nonlin2(x)

        y = x
        x = self.map3(x)
        x = self.bn3(x)
        x = self.nonlin3(x)
 
        y = self.map4(y)
        y = self.bn4(y)
        y = self.nonlin4(y)

        x = x * y

        x = self.atmap1(x)
        x = self.bnat1(x)
        x = self.nonlinat1(x)
        x = self.atmap2(x)
        
        y = x
        x = self.atmap3(x)
        x = y + x

        x = self.nonliner(x)
        x = x.view(bs,1,h,w).repeat(1,dim,1,1)
        x = x * a
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class ResNet_3G_365(nn.Module):

    def __init__(self, block, layers, num_classes=365):
        self.inplanes = 64
        super(ResNet_3G_365, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.layer_reduce = nn.Conv2d(512 * block.expansion, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(256)
        self.layer_reduce_relu = nn.ReLU(inplace=True)
        self.AtMp1 = AtMp(256)
        self.AtMp2 = AtMp(256)
        self.AtMp3 = AtMp(256)
        self.fc = nn.Linear(int(256*(256+1)/2), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
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

        x = self.layer_reduce(x)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)

        y = x 
        x = self.AtMp1(x,y)
        x = self.AtMp2(x,y)
        x = self.AtMp3(x,y)

        x = cov_forward(x)
        x = sqrt_forward(x,5)
        x = triu_forward(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_3g_365(pretrained=False, **kwargs):
    """Constructs a 3G-ResNet-18 model.
    """
    print("Run 3G-Net using ResNet-18 as backbone model on Places365")
    model = ResNet_3G_365(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_3g_365(pretrained=False, **kwargs):
    """Constructs a 3G-ResNet-34 model.
    """
    print("Run 3G-Net using ResNet-34 as backbone model on Places365")
    model = ResNet_3G_365(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet23_3g_365(pretrained=False, **kwargs):
    """Constructs a 3G-ResNet-23 model.
    """
    print("Run 3G-Net using ResNet-23 as backbone model on Places365")
    model = ResNet_3G_365(Bottleneck, [1, 2, 2, 2], **kwargs)
    return model

def resnet50_3g_365(pretrained=False, **kwargs):
    """Constructs a 3G-ResNet-50 model.
    """
    print("Run 3G-Net using ResNet-50 as backbone model on Places365")
    model = ResNet_3G_365(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_3g_365(pretrained=False, **kwargs):
    """Constructs a 3G-ResNet-101 model.
    """
    print("Run 3G-Net using ResNet-101 as backbone model on Places365")
    model = ResNet_3G_365(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152_3g_365(pretrained=False, **kwargs):
    """Constructs a 3G-ResNet-152 model.
    """
    print("Run 3G-Net using ResNet-152 as backbone model on Places365")
    model = ResNet_3G_365(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
