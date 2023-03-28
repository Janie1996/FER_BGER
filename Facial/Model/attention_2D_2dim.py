import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

class attentionblock(nn.Module):

    def __init__(self,in_channel):

        super(attentionblock,self).__init__()

        #self.num_classes=num_classes

        self.pool=nn.AdaptiveMaxPool3d(1)

        self.in_channel=in_channel

        self.conv=nn.Conv3d(self.in_channel,self.in_channel,kernel_size=(1,1,1))

        self.linear=nn.Linear(self.in_channel*2,self.in_channel*2)

        #print(self.linear)

    def forward(self,input1,input2):

        #print(input1.shape,"inpu1")

        #print(input2.shape,"input2")

        input1=self.pool(input1)

        input2=self.pool(input2)

        # input3=self.pool(input3)

        input1=self.conv(input1)

        input2 = self.conv(input2)

        # input3 = self.conv(input3)

        input1=input1.squeeze()

        input2 = input2.squeeze()

        # input3 = input3.squeeze()

        #print(input1.shape,"inpu1")

        #print(input2.shape,"inpu2")

        #print(input3.shape,"inpu3")

        a=torch.cat((input1,input2),1)

        #print(a.shape)

        a=self.linear(a)

        #print(a.shape,"attena")

        a=F.softmax(a,dim=0)

        return a


class RNN_feature(nn.Module):
    def __init__(self,in_dim,hidden_dim,classes):
        super(RNN_feature, self).__init__()
        self.lstm=nn.LSTM(in_dim,hidden_dim,1,batch_first=True)
        self.fc=nn.Linear(hidden_dim,classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # 此时可以从out中获得最终输出的状态h
        x = out[:, -1, :]
        # x = h_n[-1, :, :]
        x = self.fc(x)
        return x



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, K,inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.K=K
        self.channel= planes
        self.attenblock = attentionblock(self.channel)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        batch= int(out.shape[0]/self.K)
        clips = torch.ones([self.K,batch, out.shape[1],out.shape[2],out.shape[3]])
        for i in range(2):
            clips[i] =out[i * batch:(i + 1) * batch, :,:,:]


        out=clips.permute(1, 2, 0, 3, 4).cuda()   # batch*clips,channel,frame,w,h
        del clips

        # x1 = out.view(out.shape[0], out.shape[1], out.shape[2], out.shape[3] * out.shape[4])

        x2 = out.permute(0,1,4,3,2)
        x2 = x2.contiguous().view(x2.shape[0], x2.shape[1], x2.shape[2], x2.shape[3] * x2.shape[4])

        x3 = out.permute(0,1,3,4,2)
        x3 = x3.contiguous().view(x3.shape[0], x3.shape[1], x3.shape[2], x3.shape[3] * x3.shape[4])

        # out1 = self.conv2(x1)
        # out1 = self.bn2(out1)
        # out1 = self.relu(out1)
        # out1 = out1.view(out.shape[0], out.shape[1], out.shape[2], out.shape[3], out.shape[4])

        out2 = self.conv2(x2)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        # out2 = out2.view(out.shape[0], out.shape[1], out.shape[2], out.shape[3],out.shape[4])
        # out2 = out2.view(out.shape[0], out.shape[1], out.transpose(2, 3).shape[2],
        #                  out.transpose(2, 3).shape[3], out.transpose(2, 3).shape[4])
        # out2 = out2.transpose(2, 3)
        #  修改顺序
        out2 = out2.view(out.shape[0], out.shape[1], out.transpose(2, 4).shape[2],
                         out.transpose(2, 4).shape[3], out.transpose(2, 4).shape[4])
        out2 = out2.transpose(2, 4)

        out3 = self.conv2(x3)
        out3 = self.bn2(out3)
        out3 = self.relu(out3)


        # out3 = out3.view(out.shape[0], out.shape[1], out.transpose(2, 4).shape[2],
        #                  out.transpose(2, 4).shape[3], out.transpose(2, 4).shape[4])
        # out3 = out3.transpose(2, 4)

        out3 = out3.view(out.shape[0], out.shape[1], out.permute(0,1,3,4,2).shape[2],
                         out.permute(0,1,3,4,2).shape[3], out.permute(0,1,3,4,2).shape[4])
        out3 = out3.permute(0,1,4,2,3)

        out = out2 + out3

        # a = self.attenblock(out2, out3)
        #
        # a2, a3 = a.chunk(2, dim=1)  # batch, channel
        #
        # output1 =  out2.permute(2, 3, 4, 0, 1) * a2 + out3.permute(2, 3, 4, 0, 1) * a3
        #
        # out = output1.permute(3, 4, 0, 1, 2)

        #out = out1 + out2 + out3



        input = out[:, :, 0, :, :].unsqueeze(0)
        for j in range(1, self.K):
            input = torch.cat([input, out[:, :, j, :, :].unsqueeze(0)], 0)
        out = input.view(-1, out.shape[1], out.shape[3], out.shape[4]).cuda()
        del input


        # out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers,batch,C,K, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.batch = batch
        self.K = K
        self.C = C
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        self.rnn_size = 512 * block.expansion
        self.rnn = RNN_feature(self.rnn_size, 1024, num_classes)
        self.rnn1=RNN_feature(num_classes,512,num_classes)
        self.dropout=nn.Dropout(p=0.5)
        self.avg_1d_pool=nn.AdaptiveAvgPool1d(1)
        self.soft=nn.Softmax(1)
        
        self.num_classes=num_classes
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.K,self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.K,self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):  # [batch*clips, chanel, frame,h,w]

        #x=x.permute(2,0,1,3,4)
        input=x[:,:,0,:,:].unsqueeze(0)
        for j in range(1,self.K):
            input = torch.cat([input, x[:,:,j,:,:].unsqueeze(0)], 0)
        x = input.view(-1, 3, 224, 224).cuda()
        del input
        #
        # clips = torch.ones([2, 40, 3,224,224])
        # for i in range(2):
        #     clips[i] =input[i * 40:(i + 1) * 40, :,:,:]
        #
        # clips=clips.permute(1, 2, 0, 3, 4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # clips = torch.ones([2, 40, 3,224,224])
        # for i in range(2):
        #     clips[i] =input[i * 40:(i + 1) * 40, :,:,:]
        #
        # clips=clips.permute(1, 2, 0, 3, 4)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        batch=int(x.shape[0]/self.K)

        clips = torch.ones([self.K,batch, x.shape[1],x.shape[2],x.shape[3]])
        for i in range(self.K):
            clips[i] =x[i * batch:(i + 1) * batch, :,:,:]

        x=clips.permute(1, 2, 0, 3, 4).cuda()
        del clips

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.fc(x)

        batch=int(x.shape[0]/self.C)
        clips = torch.ones([self.C, batch, self.rnn_size])

        for i in range(self.C):
            clips[i] = x[i * batch:(i + 1) * batch, :]
        clips = clips.transpose(0, 1).cuda()

        end = self.rnn(clips)

        out=self.dropout(x)
        out=self.fc1(out)

        out=self.soft(out)

        probs=out.view(self.C,-1,self.num_classes).permute(1,2,0)
        probs=self.avg_1d_pool(probs).squeeze(2)

        clips=out.view(self.C,-1,self.num_classes).permute(1,0,2)
        out=self.rnn1(clips)
        del clips

        return probs,out,end,0.2*probs+0.8*out  ###


def _resnet(arch, block, layers, batch,C,K,num_classes,pretrained, progress, **kwargs):
    model = ResNet(block, layers, batch,C,K,num_classes,**kwargs)
    return model


def resnet18(batch,C,K,num_classes,pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], batch,C,K,num_classes,pretrained, progress,
                   **kwargs)


def resnet34(batch,C,K,num_classes,pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], batch,C,K,num_classes,pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


if __name__ == "__main__":
    inputs = torch.rand(10, 3, 2, 224, 224).cuda()

    net=resnet18(batch=2, C=5, K=2, num_classes=7).cuda()

    outputs = net(inputs)

    print(outputs.size())