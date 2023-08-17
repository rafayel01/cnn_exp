from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.

    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss
    


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
from torch.utils.data.sampler import SubsetRandomSampler

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
valset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
valid_size = 0.2
num_train = len(trainset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
np.random.seed(42)
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                          num_workers=2, pin_memory=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=valid_sampler,
                                          num_workers=2, pin_memory=True)

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
          for data_t, target_t in (valloader):
              data_t, target_t = data_t.to(device), target_t.to(device)
              outputs_t = model(data_t)
              loss_t = criterion(outputs_t, target_t)
              batch_loss += loss_t.item()
              _,pred_t = torch.max(outputs_t, dim=1)
              correct_t += torch.sum(pred_t==target_t).item()
              total_t += target_t.size(0)
          val_acc.append(100 * correct_t/total_t)
          val_loss.append(batch_loss/len(valloader))
          network_learned = batch_loss < valid_loss_min
          print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')
            
          
          if network_learned:
              valid_loss_min = batch_loss
              torch.save(model.state_dict(), f'/home/rafayel.veziryan/cnn_exp/results/cifar10_2/{model._get_name()}_bst.pt')
              print('Improvement-Detected, save-model')
          model.train()
  test_loss = 0
  total_test_t=0
  correct_test_t=0
  with torch.no_grad():
    model.eval()
    for data_t, target_t in (testloader):
        data_t, target_t = data_t.to(device), target_t.to(device)
        outputs_t = model(data_t)
        loss_t = criterion(outputs_t, target_t)
        test_loss += loss_t.item()
        _,pred_t = torch.max(outputs_t, dim=1)
        correct_test_t += torch.sum(pred_t==target_t).item()
        total_test_t += target_t.size(0)
    test_acc = (100 * correct_test_t/total_test_t)
    print(f'Test loss: {(test_loss):.4f}, Test acc: {(100 * correct_test_t/total_test_t):.4f}\n')
  with open(f"/home/rafayel.veziryan/cnn_exp/results/cifar10_2/{model._get_name()}_test_results.txt", 'w') as f:
      f.write(f'Test loss: {(test_loss):.4f}, Test acc: {(100 * correct_test_t/total_test_t):.4f}\n')
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
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
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
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
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
models = (model_original, ) #, model_cable_eq,  model_cable_eq_2, model_cable_eq_3, model_cable_eq_4, model_cable_eq_5)

for model in models:
    print(f"{model._get_name()}: {parameter_count(model)}")



for model in models:
    valid_loss_min = np.Inf
    n_epochs = 100
    max_lr = 0.1
    grad_clip = 0.1
    weight_decay = 1e-4
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=weight_decay, momentum=0.9)
    optimizer = Lookahead(optimizer) #, la_steps=args.la_steps, la_alpha=args.la_alpha)
    criterion = F.cross_entropy
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=n_epochs, steps_per_epoch=len(trainloader))
    cable_eq_tr_loss, cable_eq_tr_acc, cable_eq_v_loss, cable_eq_v_acc = training(model, n_epochs, optimizer, criterion, sched)
    torch.save(model.state_dict(), f'/home/rafayel.veziryan/cnn_exp/results/cifar10_2/{model._get_name()}_best.pt')
    model_cable_eq_dict ={}
    model_cable_eq_dict['train_loss']=cable_eq_tr_loss
    model_cable_eq_dict['train_acc']=cable_eq_tr_acc
    model_cable_eq_dict['test_loss']=cable_eq_v_loss
    model_cable_eq_dict['test_acc']=cable_eq_v_acc
    with open(f'/home/rafayel.veziryan/cnn_exp/results/cifar10_2/{str(model._get_name())}_bn.pkl', 'wb') as fp:
        pickle.dump(model_cable_eq_dict, fp)
        print('dictionary saved successfully to file')
