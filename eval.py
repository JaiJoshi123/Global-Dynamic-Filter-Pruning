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

import matplotlib.pyplot as plt
import numpy as np

from nets.resnet import preresnet

data_path = "/home/hice1/jjoshi48/scratch/eml_project/datasets"
pretrained_model_dir = "/home/hice1/jjoshi48/scratch/eml_project/pretrained_models"
pruned_model_dir = "/home/hice1/jjoshi48/scratch/eml_project/pruned_models"

beta = 0.09
dataset_name = "cifar10"
depth = 20

model_name = f"resnet{depth}_{dataset_name}_updated_{beta}"
results_dir = f"/home/hice1/jjoshi48/scratch/eml_project/results/{model_name}"
data_path = f"{pruned_model_dir}/{model_name}.pt"

def plotting_function(path_name):
  data = torch.load(path_name, map_location=torch.device('cpu'))
  test_acc = data["test_accuracies"]
  test_loss = data["test_loss"]

  epochs = [i+1 for i in range(len(test_acc))]

  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
  ax1.plot(epochs, test_acc)
  ax1.set_title("Validation Accuracies")
  ax1.set_xlabel("Epochs")
  ax1.grid(visible=True)
  ax1.set_ylabel("Validation Accuracies")

  ax2.plot(epochs, test_loss)
  ax2.set_title("Validation Loss")
  ax2.set_xlabel("Epochs")
  ax2.set_ylabel("Validation Loss")
  ax2.grid(visible=True)
  plt.savefig(results_dir+"_results.png")

plotting_function(data_path)