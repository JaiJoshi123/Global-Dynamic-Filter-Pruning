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

torch.cuda.empty_cache()
# from pretrain_model import final_train, test

data_path = "/home/hice1/jjoshi48/scratch/eml_project/datasets"
pretrained_model_dir = "/home/hice1/jjoshi48/scratch/eml_project/pretrained_models"
pruned_model_dir = "/home/hice1/jjoshi48/scratch/eml_project/pruned_models"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device: ",device)

# from flopth import flopth
from ptflops import get_model_complexity_info

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

def final_train(model , epochs, trainloader, testloader, path_name="updated"):
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

  prune_params = []
  for name, module in model.named_modules():
      if isinstance(module, torch.nn.Conv2d) and "downsample" not in name:
          prune_params.append((module, "weight"))

  t = 0
  d = 0
  for m,w in prune_params:
      t+=100. * float(torch.sum(m.weight == 0))
      d+= m.weight.nelement()

  torch.save({"train_accuracies": train_acc,
              "train_loss": train_loss,
              "test_accuracies": test_acc,
              "test_loss": test_loss,
              "Global_sparsity": t/d,
              "Model_actual_size_percentage": 100-t/d,
              "model": best_model_dict
              },
            pruned_model_dir+f"/resnet{depth}_{dataset_name}_{path_name}.pt")
  print('Finished Training')

def update_mask(model, beta):
    # calculate the saliency scores of the filters
    saliency_scores = []
    for id, data in enumerate(model.named_parameters()):
      # print(id)
      name, param = data
      # print(name)
      if "conv" in name:
        param.grad.requires_grad = True
        loss_gradients = param.grad
        filter_params = param.data
        loss_gradients.requires_grad = True
        filter_params.requires_grad = True
        # filter_params.retain_grad()
        # print(loss_gradients.requires_grad)
        # print(filter_params.requires_grad)
        # print(loss_gradients.shape)
        # print(filter_params.shape)

        n, c, k1, k2 = filter_params.shape
        for i in range(n):
          score = torch.abs(torch.sum(loss_gradients[i] * filter_params[i]))
          # print(score.item())
          saliency_scores.append((score.item(), name[:-12], i))


    # Sort indices based on saliency scores in descending order
    saliency_scores = sorted(saliency_scores, reverse=True)
    saliency_scores = saliency_scores[:int(beta*len(saliency_scores))]
    saliency_scores_dict = {}
    for score, layer, id in saliency_scores:
      if layer in saliency_scores_dict:
        saliency_scores_dict[layer].append(id)
      else:
        saliency_scores_dict[layer] = [id]
    
    # print(saliency_scores_dict)
    # Update mask based on top-beta indices
    for name, module in model.named_buffers():
      if 'conv' in name:
        # print(f"{name[:-12]}.weight_mask")
        # print(f"{name[:-12]}.weight")
        n, c, k1, k2 = module.shape
        #find the module to prune
        module_to_prune = ''
        for module_name, module in model.named_modules():
          if "conv" in module_name:
            # print(module_name)
            if name[:-12] == module_name:
              module_to_prune = module
              break
        # print("Module to prune:", module_to_prune)
        # print("Name of layer: ", name[:-12])
        if name[:-12] in saliency_scores_dict:
          # update mask for individual wts
        #   print("Updating the mask of individual wts")
          mask = torch.tensor([])
          for id in range(n):
            if id in saliency_scores_dict[name[:-12]]:
              mask = torch.cat((mask, torch.ones(1, c, k1, k2)))
            else:
              mask = torch.cat((mask, torch.zeros(1, c, k1, k2)))
          #update mask for the layer
          mask = mask.to(device)
          module_to_prune.register_buffer(name="weight_mask", tensor=mask, persistent=True)
        
        # else:
        #   # prune whole layer
        #   module_to_prune.register_buffer(name="weight_mask", tensor=torch.zeros(n, c, k1, k2).to(device), persistent=True)

def global_dynamic_pruning(training_data, updated_model, learning_rate,
                           threshold_update_mask, max_iterations, beta):

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(updated_model.parameters(), lr=learning_rate, momentum=0.9)

    # Algorithm iterations
    for t in tqdm(range(max_iterations)):

      if t==20:
        threshold_update_mask = 1
      
      # print(f"Starting Iteration {t}")
      # 3: Forward Pass
      # Choose a minibatch from training_data
      dataloader_iterator = iter(training_data)
      x, y = next(dataloader_iterator)
      x, y = x.to(device), y.to(device)
      # x,y = data
      # print(x,y)

      # Zero gradients
      optimizer.zero_grad()

      # Forward pass
    #   print("Is cuda x: ", x.is_cuda)
    #   print("Is cuda y: ", y.is_cuda)
    #   print("Is model on cuda: ", next(updated_model.parameters()).is_cuda)
      outputs = updated_model(x)
      # print(outputs)

      # Compute loss
      loss = criterion(outputs, y)

      # 4: Backward Pass
      # Compute gradient
      loss.backward()

      # 5: Update
      if (t + 1) % threshold_update_mask == 0: 
          update_mask(updated_model, beta)

      # Perform optimization step
      optimizer.step()

# Call the global_dynamic_pruning function
if __name__ == "__main__":
    beta = 0.7
    dataset_name = "cifar10"
    depth = 20

    print(f"Start Pruning the models on {dataset_name} with depth={depth} and beta={beta}")

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

    original_model_path = f"{pretrained_model_dir}/resnet{depth}_{dataset_name}_original.pt"
    if os.path.exists(original_model_path):
        print("Model Already Exists! Copying into updated model")
        updated_model = preresnet(depth=depth).to(device)
        updated_model.load_state_dict(torch.load(original_model_path, map_location=device)["model"])
        updated_model = updated_model.to(device)
        updated_model.train()

        print("Started Pruning the Model.")

        for name, module in updated_model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and "downsample" not in name:
                prune.identity(module, name='weight')

        global_dynamic_pruning(trainloader, updated_model, learning_rate=0.001, threshold_update_mask=2, max_iterations=30, beta=beta)

        print("Pruning completed.")
        prune_params = []
        for name, module in updated_model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and "downsample" not in name:
                prune_params.append((module, "weight"))

        t = 0
        d = 0
        for m,w in prune_params:
            t+=100. * float(torch.sum(m.weight == 0))
            d+= m.weight.nelement()

        print("Global sparsity of updated model: ", t/d)
        print(f"Model is shrinked to: {100-t/d}%")
        
        # print("Updated Model complexity: ")
        # macs, params = get_model_complexity_info(updated_model, (3, 32, 32), as_strings=True,
        #                                    print_per_layer_stat=True, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # for name, module in updated_model.named_modules():
        #     if isinstance(module, torch.nn.Conv2d) and "downsample" not in name:
        #         prune.remove(module, name='weight')
        

        # macs, params = get_model_complexity_info(updated_model, (3, 32, 32), as_strings=True,
        #                                    print_per_layer_stat=True, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # Start testing
        print("Initial Testing Result:")
        teacc, teloss = test(updated_model, device, testloader)

        #Start Fine tuning
        final_train(model=updated_model, epochs=50, trainloader=trainloader, testloader=testloader, path_name=f"updated_{beta}")
        print("Finished Finetuning")
    else:
        print("First pretrain the model!")