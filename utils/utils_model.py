import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import datetime


class SimpleNet(nn.Module):
    def __init__(self, name=None, created_time=None):
        super(SimpleNet, self).__init__()
        self.created_time = created_time
        self.name=name

    def train_vis(self, vis, epoch, acc, loss=None, eid='main', is_poisoned=False, name=None):
        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        vis.line(X=np.array([epoch]), Y=np.array([acc]), name=name, win='train_acc_{0}'.format(self.created_time), env=eid,
                                update='append' if vis.win_exists('train_acc_{0}'.format(self.created_time), env=eid) else None,
                                opts=dict(showlegend=True, title='Train Accuracy_{0}'.format(self.created_time),
                                          width=700, height=400))
        if loss is not None:
            vis.line(X=np.array([epoch]), Y=np.array([loss]), name=name, env=eid,
                                     win='train_loss_{0}'.format(self.created_time),
                                     update='append' if vis.win_exists('train_loss_{0}'.format(self.created_time), env=eid) else None,
                                     opts=dict(showlegend=True, title='Train Loss_{0}'.format(self.created_time), width=700, height=400))
        return

    def train_batch_vis(self, vis, epoch, data_len, batch, loss, eid='main', name=None, win='train_batch_loss', is_poisoned=False):
        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        else:
            name = name + '_poisoned' if is_poisoned else name

        vis.line(X=np.array([(epoch-1)*data_len+batch]), Y=np.array([loss]),
                                 env=eid,
                                 name=f'{name}' if name is not None else self.name, win=f'{win}_{self.created_time}',
                                 update='append' if vis.win_exists(f'{win}_{self.created_time}', env=eid) else None,
                                 opts=dict(showlegend=True, width=700, height=400, title='Train Batch loss_{0}'.format(self.created_time)))
    def track_distance_batch_vis(self,vis, epoch, data_len, batch, distance_to_global_model,eid,name=None,is_poisoned=False):
        x= (epoch-1)*data_len+batch+1

        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        else:
            name = name + '_poisoned' if is_poisoned else name


        vis.line(Y=np.array([distance_to_global_model]), X=np.array([x]),
                 win=f"global_dist_{self.created_time}",
                 env=eid,
                 name=f'Model_{name}',
                 update='append' if
                 vis.win_exists(f"global_dist_{self.created_time}",
                                env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Distance to Global {self.created_time}",
                           width=700, height=400))
    def weight_vis(self,vis,epoch,weight, eid, name,is_poisoned=False):
        name = str(name) + '_poisoned' if is_poisoned else name
        vis.line(Y=np.array([weight]), X=np.array([epoch]),
                 win=f"Aggregation_Weight_{self.created_time}",
                 env=eid,
                 name=f'Model_{name}',
                 update='append' if
                 vis.win_exists(f"Aggregation_Weight_{self.created_time}",
                                env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Aggregation Weight {self.created_time}",
                           width=700, height=400))

    def alpha_vis(self,vis,epoch,alpha, eid, name,is_poisoned=False):
        name = str(name) + '_poisoned' if is_poisoned else name
        vis.line(Y=np.array([alpha]), X=np.array([epoch]),
                 win=f"FG_Alpha_{self.created_time}",
                 env=eid,
                 name=f'Model_{name}',
                 update='append' if
                 vis.win_exists(f"FG_Alpha_{self.created_time}",
                                env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"FG Alpha {self.created_time}",
                           width=700, height=400))

    def trigger_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key, trigger_name, trigger_value):
        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"poison_triggerweight_vis_acc_{self.created_time}",
                 env=eid,
                 name=f'{agent_name_key}_[{trigger_name}]_{trigger_value}',
                 update='append' if vis.win_exists(f"poison_trigger_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Backdoor Trigger Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"poison_trigger_loss_{self.created_time}",
                     env=eid,
                     name=f'{agent_name_key}_[{trigger_name}]_{trigger_value}',
                     update='append' if vis.win_exists(f"poison_trigger_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Backdoor Trigger Test Loss_{self.created_time}",
                               width=700, height=400))

    def trigger_agent_test_vis(self, vis, epoch, acc, loss, eid, name):
        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"poison_state_trigger_acc_{self.created_time}",
                 env=eid,
                 name=f'{name}',
                 update='append' if vis.win_exists(f"poison_state_trigger_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Backdoor State Trigger Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"poison_state_trigger_loss_{self.created_time}",
                     env=eid,
                     name=f'{name}',
                     update='append' if vis.win_exists(f"poison_state_trigger_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Backdoor State Trigger Test Loss_{self.created_time}",
                               width=700, height=400))


    def poison_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name= agent_name_key
        # name= f'Model_{name}'

        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"poison_test_acc_{self.created_time}",
                 env=eid,
                 name=name,
                 update='append' if vis.win_exists(f"poison_test_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Backdoor Task Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"poison_loss_acc_{self.created_time}",
                     env=eid,
                     name=name,
                     update='append' if vis.win_exists(f"poison_loss_acc_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Backdoor Task Test Loss_{self.created_time}",
                               width=700, height=400))

    def additional_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name = agent_name_key
        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"additional_test_acc_{self.created_time}",
                 env=eid,
                 name=name,
                 update='append' if vis.win_exists(f"additional_test_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Additional Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"additional_test_loss_{self.created_time}",
                     env=eid,
                     name=name,
                     update='append' if vis.win_exists(f"additional_test_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Additional Test Loss_{self.created_time}",
                               width=700, height=400))


    def test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name= agent_name_key
        # name= f'Model_{name}'

        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"test_acc_{self.created_time}",
                 env=eid,
                 name=name,
                 update='append' if vis.win_exists(f"test_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Main Task Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"test_loss_{self.created_time}",
                     env=eid,
                     name=name,
                     update='append' if vis.win_exists(f"test_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Main Task Test Loss_{self.created_time}",
                               width=700, height=400))


    def save_stats(self, epoch, loss, acc):
        self.stats['epoch'].append(epoch)
        self.stats['loss'].append(loss)
        self.stats['acc'].append(acc)

    def copy_params(self, state_dict, coefficient_transfer=100):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                #random_tensor = (torch.cuda.FloatTensor(shape).random_(0, 100) <= coefficient_transfer).type(torch.cuda.FloatTensor)
                # negative_tensor = (random_tensor*-1)+1
                # own_state[name].copy_(param)
                own_state[name].copy_(param.clone())

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(SimpleNet):
    def __init__(self, block, num_blocks, num_classes=10, name=None, created_time=None):
        super(ResNet, self).__init__(name, created_time)
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # for SDTdata
        # return F.softmax(out, dim=1)
        # for regular output
        return out


class MnistNet(SimpleNet):
    def __init__(self, name=None, created_time=None):
        super(MnistNet, self).__init__(f'{name}_Simple', created_time)

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        # self.fc2 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # in_features = 28 * 28
        # x = x.view(-1, in_features)
        # x = self.fc2(x)

        # normal return:
        return F.log_softmax(x, dim=1)
        # soft max is used for generate SDT data
        # return F.softmax(x, dim=1)

class BasicBlockTiny(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
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

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetTiny(SimpleNet):

    def __init__(self, block, layers,name=None, created_time=None, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__(name, created_time)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

        # change the model to fit our tiny-imagenet-200 (200 classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, 200)


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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

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
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x