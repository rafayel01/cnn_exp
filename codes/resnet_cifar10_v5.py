import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
# %matplotlib inline
from torchvision import transforms
from torchsummary import summary
import pickle

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def parameter_count(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
# # Ploting results after my layers
# filter = torch.tensor([[0., 1. , 0], 
#                        [0., -2., 0.],
#                        [0., 1. , 0]])
# filter2 = torch.tensor([[0., 0. , 0], 
#                        [1., -2., 1.],
#                        [0., 0. , 0]])
# filter3 = torch.tensor([[0., 0. , 0], 
#                        [1., -2., 1.],
#                        [0., 0. , 0]])
# f = filter.expand(3,3,3,3)
# f2 = filter2.expand(3,3,3,3)
# img = next(iter(trainloader))[0][0]
# after_layer = F.conv2d(img, f, stride=1, padding=1) + F.conv2d(img, f2, stride=1, padding=1)
# imshow(after_layer)

transform_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),inplace=True)])
transform_test = transforms.Compose([
    transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),inplace=True)
])
batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
device = get_default_device()
print(device)

def parameter_count(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def training(model, n_epochs, optimizer, criterion, sched):
  model.train
  valid_loss_min = np.Inf
  val_loss = []
  val_acc = []
  train_loss = []
  train_acc = []
  total_step = len(trainloader)
  for epoch in range(1, n_epochs+1):
      running_loss = 0.0
      correct = 0
      total=0
      print(f'Epoch {epoch}\n')
      for batch_idx, (data_, target_) in enumerate(trainloader):
          data_, target_ = data_.to(device), target_.to(device)
          
          outputs = model(data_)
          loss = criterion(outputs, target_)
          loss.backward()
          #nn.utils.clip_grad_value_(model.parameters(), grad_clip)
          optimizer.step()
          optimizer.zero_grad()
          sched.step()

          running_loss += loss.item()
          pred = torch.argmax(outputs, dim=1)
          correct += torch.sum(pred==target_).item()
          total += target_.size(0)
          if (batch_idx) % 1000 == 0:
              print (f"Epoch [{epoch}/{n_epochs}], Loss: {round(loss.item(), 4)}, LR: {round(optimizer.param_groups[0]['lr'], 4)}" )
      
      train_acc.append(100 * correct / total)
      train_loss.append(running_loss/total_step)
      print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
      batch_loss = 0
      total_t=0
      correct_t=0
      with torch.no_grad():
          model.eval()
          for data_t, target_t in (testloader):
              data_t, target_t = data_t.to(device), target_t.to(device)
              outputs_t = model(data_t)
              loss_t = criterion(outputs_t, target_t)
              batch_loss += loss_t.item()
              _,pred_t = torch.max(outputs_t, dim=1)
              correct_t += torch.sum(pred_t==target_t).item()
              total_t += target_t.size(0)
          val_acc.append(100 * correct_t/total_t)
          val_loss.append(batch_loss/len(testloader))
          network_learned = batch_loss < valid_loss_min
          print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

          
          if network_learned:
              valid_loss_min = batch_loss
              torch.save(model.state_dict(), f'/home/rafayel.veziryan/cnn_exp/results/cifar10/with_relu_bn/lr_0.5/{model._get_name()}_bst.pt')
              print('Improvement-Detected, save-model')
      model.train()
  return train_loss, train_acc, val_loss, val_acc

from typing import Type
class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()

        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = my_layer(in_channels=in_channels,
                              out_channels=out_channels,
                              stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = my_layer(in_channels=out_channels,
                              out_channels=out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return  out

class ResNet18(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 10
    ) -> None:
        super(ResNet18, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 3
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=3, 
            stride=2,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 3,  32, layers[0], downsample=1)
        self.layer2 = self._make_layer(block, 32,  64, layers[1], stride=2, downsample=1)
        self.layer3 = self._make_layer(block, 64, 128, layers[2], stride=2, downsample=1)
        self.conv_last = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3, 
            stride=2,
            padding=1,
            bias=False
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(256*self.expansion, num_classes),
                                nn.LogSoftmax(dim=1))
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1,
        downsample = None
    ) -> nn.Sequential:
        if downsample:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            print(f"in channels: {self.in_channels}")
            print(f"make layer out channels: {out_channels}")
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
            nn.BatchNorm2d(out_channels * self.expansion),
            )
            
        layers = []
        layers.append(
            block(
                in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        
        self.in_channels = out_channels * self.expansion
        print(f"self expansion: {self.expansion}")
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv_last(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class my_layer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        self.stride = stride
        filter1 = torch.Tensor([[[0,  1,  0], 
                                [0, -2,  0],
                                [0,  1,  0]]])
        filter2 = torch.Tensor([[[0,  0,  0], 
                                [1, -2,  1],
                                [0,  0,  0]]])
        filter3 = torch.Tensor([[[0,  0,  0], 
                                 [0,  1,  0],
                                 [0,  0,  0]]])
        self.register_buffer("filter1", filter1)
        self.register_buffer("filter2", filter2)
        self.register_buffer("filter3", filter3)
        self.weight1_1 = nn.Parameter(torch.Tensor(out_channels, in_channels, 1))
        self.weight1_2 = nn.Parameter(torch.Tensor(out_channels, in_channels, 1))
        self.weight1_3 = nn.Parameter(torch.Tensor(out_channels, in_channels, 1))
        #self.bias = nn.Parameter(torch.Tensor(1))
        nn.init.xavier_normal_(self.weight1_1)
        nn.init.xavier_normal_(self.weight1_2)
        nn.init.xavier_normal_(self.weight1_3)
        self.bn01 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.filter_1 = filter1.to(device)
        self.filter_2 = filter2.to(device)
        self.filter_3 = filter3.to(device)


    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        self.kernel1_1 = torch.einsum("ijk, klm -> ijlm", self.weight1_1, self.filter_1)
        self.kernel1_2 = torch.einsum("ijk, klm -> ijlm", self.weight1_2, self.filter_2)
        self.kernel1_3 = torch.einsum("ijk, klm -> ijlm", self.weight1_3, self.filter_3)
        x = F.conv2d(x, weight=self.kernel1_1+self.kernel1_2+self.kernel1_3, stride=self.stride, padding=1)
        x = self.bn01(x)
        x = self.relu(x)
        return x


class ResNet18_with_cable_eq(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 10
    ) -> None:
        super(ResNet18_with_cable_eq, self).__init__()
    
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.mylayer_1 = my_layer(img_channels=img_channels)
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512*self.expansion, num_classes),
                                nn.LogSoftmax(dim=1))
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mylayer_1(x)
        print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18_with_cable_eq_2(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 10
    ) -> None:
        super(ResNet18_with_cable_eq_2, self).__init__()
        # Initialazing kernels
        
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.mylayer_1 = my_layer(img_channels=img_channels)
        self.mylayer_2 = my_layer(img_channels=img_channels)
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512*self.expansion, num_classes),
                                nn.LogSoftmax(dim=1))
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mylayer_1(x)
        x = self.mylayer_2(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18_with_cable_eq_3(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 10
    ) -> None:
        super(ResNet18_with_cable_eq_3, self).__init__()
        
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.mylayer_1 = my_layer(img_channels=img_channels)
        self.mylayer_2 = my_layer(img_channels=img_channels)
        self.mylayer_3 = my_layer(img_channels=img_channels)
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512*self.expansion, num_classes),
                                nn.LogSoftmax(dim=1))
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mylayer_1(x)
        x = self.mylayer_2(x)
        x = self.mylayer_3(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18_with_cable_eq_4(nn.Module):
    def __init__(
        self,
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 10
    ) -> None:
        super(ResNet18_with_cable_eq_4, self).__init__()
    
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock`
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.mylayer_1 = my_layer(img_channels=img_channels)
        self.mylayer_2 = my_layer(img_channels=img_channels)
        self.mylayer_3 = my_layer(img_channels=img_channels)
        self.mylayer_4 = my_layer(img_channels=img_channels)
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512*self.expansion, num_classes),
                                nn.LogSoftmax(dim=1))
    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mylayer_1(x)
        x = self.mylayer_2(x)
        x = self.mylayer_3(x)
        x = self.mylayer_4(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18_with_cable_eq_5(nn.Module):
    def __init__(
        self,
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 10
    ) -> None:
        super(ResNet18_with_cable_eq_5, self).__init__()
        
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock`
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.mylayer_1 = my_layer(img_channels=img_channels)
        self.mylayer_2 = my_layer(img_channels=img_channels)
        self.mylayer_3 = my_layer(img_channels=img_channels)
        self.mylayer_4 = my_layer(img_channels=img_channels)
        self.mylayer_5 = my_layer(img_channels=img_channels)
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512*self.expansion, num_classes),
                                nn.LogSoftmax(dim=1))
    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mylayer_1(x)
        x = self.mylayer_2(x)
        x = self.mylayer_3(x)
        x = self.mylayer_4(x)
        x = self.mylayer_5(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model_original = ResNet18(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
#model_cable_eq = ResNet18_with_cable_eq(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
#model_cable_eq_2 = ResNet18_with_cable_eq_2(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
##model_cable_eq_3 = ResNet18_with_cable_eq_3(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
#model_cable_eq_4 = ResNet18_with_cable_eq_4(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
#model_cable_eq_5 = ResNet18_with_cable_eq_5(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
models = (model_original, ) #, model_cable_eq_2, model_cable_eq_3, model_cable_eq_4, model_cable_eq_5)

for model in models:
    print(f"{model._get_name()}: {parameter_count(model)}")



for model in models:
    valid_loss_min = np.Inf
    n_epochs = 100
    max_lr = 0.5
    grad_clip = 0.1
    weight_decay = 1e-4
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.5, weight_decay=weight_decay, momentum=0.9)
    criterion = F.cross_entropy
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=n_epochs, steps_per_epoch=len(trainloader))
    cable_eq_tr_loss, cable_eq_tr_acc, cable_eq_v_loss, cable_eq_v_acc = training(model, n_epochs, optimizer, criterion, sched)
    #torch.save(model.state_dict(), f'/home/rafayel.veziryan/cnn_exp/results/cifar10/with_relu_bn/lr_0.5/{model._get_name()}_best.pt')
    model_cable_eq_dict ={}
    model_cable_eq_dict['train_loss']=cable_eq_tr_loss
    model_cable_eq_dict['train_acc']=cable_eq_tr_acc
    model_cable_eq_dict['test_loss']=cable_eq_v_loss
    model_cable_eq_dict['test_acc']=cable_eq_v_acc
    #with open(f'/home/rafayel.veziryan/cnn_exp/results/cifar10/with_relu_bn/lr_0.5/{str(model._get_name())}_bn.pkl', 'wb') as fp:
    #    pickle.dump(model_cable_eq_dict, fp)
    #    print('dictionary saved successfully to file')




# Printing images, before and after my layers
'''
imgs = next(iter(trainloader))
images, labels = imgs[0][:4], imgs[1][:4]
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
classes = trainset.classes
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

model_cable_eq = model_cable_eq.cpu()

after_f1 = model_cable_eq.mylayer_1(images)
imshow(torchvision.utils.make_grid(after_f1))
# print labels
classes = trainset.classes
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
#####################
'''
