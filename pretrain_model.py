from __future__ import absolute_import
import math

import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import torch.nn.utils.prune as prune
import copy
import os

from nets.resnet import preresnet
from ptflops import get_model_complexity_info

data_path = "/home/hice1/jjoshi48/scratch/eml_project/datasets"
pretrained_model_dir = "/home/hice1/jjoshi48/scratch/eml_project/pretrained_models"
pruned_model_dir = "/home/hice1/jjoshi48/scratch/eml_project/pruned_models"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(dry_run,log_interval, model, device, train_loader, optimizer, epoch):
  model.train()
  print("Epoch: ",epoch)
  total_train_loss = 0
  counter = 0
  correct=0
  for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)
      train_loss = F.cross_entropy(output, target)
      total_train_loss += train_loss.item()
      train_loss.backward()
      optimizer.step()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()
      counter+=1
      # if batch_idx % log_interval == 0:
      #     # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
      #     #     epoch, batch_idx * len(data), len(train_loader.dataset),
      #     #     100. * batch_idx / len(train_loader), loss.item()))
      #     if dry_run:
      #         break
  # train_loss /= len(train_loader.dataset)
  train_acc = 100. * correct / len(train_loader.dataset)
  return train_acc, float(total_train_loss/counter)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    test_acc = 0
    print("\nStart Testing")
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_acc, float(test_loss)

def final_train(model , epochs, trainloader, testloader, path_name="original"):
  print("\nStart Training")
  log_interval = 10
  dry_run = False

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=epochs, steps_per_epoch=len(trainloader))
  train_acc, train_loss, test_acc, test_loss = [],[],[],[]

  best_acc = 0
  best_model_dict = ''
  for epoch in range(1, epochs+1):
    tracc, trloss = train(dry_run, log_interval, model, device, trainloader, optimizer, epoch)
    teacc, teloss = test(model, device, testloader)
    train_acc.append(tracc)
    train_loss.append(trloss)
    test_acc.append(teacc)
    test_loss.append(teloss)
    scheduler.step()

    if teacc > best_acc:
      best_acc = teacc
      best_model_dict = copy.deepcopy(model.state_dict())

  torch.save({"train_accuracies": train_acc,
              "train_loss": train_loss,
              "test_accuracies": test_acc,
              "test_loss": test_loss,
              "model": best_model_dict
              },
            pretrained_model_dir+f"/resnet{depth}_{dataset_name}_{path_name}.pt")
  print('Finished Training')

if __name__ == "__main__":
  
  dataset_name = "cifar10"
  depth = 20
  imagenet_train_data_dir = ""
  imagenet_test_data_dir = ""

  print(f"Start Pretraining the models on {dataset_name} with depth={depth}")

  if dataset_name == "cifar10":
    transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  elif dataset_name=="cifar100":
    transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16

    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)
  elif dataset_name=="imagenet":
    transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16
    
    trainset = torchvision.datasets.ImageNet(imagent_train_data_dir)
    trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4,
                                          shuffle=True)

    testset = torchvision.datasets.ImageNet(imagent_test_data_dir)
    testloader = torch.utils.data.DataLoader(testset,
                                          batch_size=4,
                                          shuffle=True)
  model = preresnet(depth=depth).to(device)

  final_train(model=model, epochs=30, trainloader=trainloader, testloader=testloader, path_name="original")