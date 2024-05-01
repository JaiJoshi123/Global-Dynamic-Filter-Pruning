# Accelerating Convolutional Networks via Global & Dynamic Filter Pruning

## Overview

Welcome to the repository containing the code implementation of "Accelerating Convolutional Networks via Global & Dynamic Filter Pruning." This paper presents an innovative approach to accelerate convolutional neural networks by globally and dynamically pruning redundant filters. The code provided here enables you to reproduce the experiments and results described in the paper. The link to the paper can be found below:

[Accelerating Convolutional Networks via Global & Dynamic Filter Pruning](https://www.ijcai.org/proceedings/2018/0336.pdf)

## Prerequisites

Before running the scripts, ensure you have the following installed:

- Python >= 3.8.x
- PyTorch (install instructions [here](https://pytorch.org/get-started/locally/))
- CUDA (if you plan to use GPU for training)
- Additional dependencies listed in `requirements.txt`

You can install dependencies using pip:

```bash
pip install -r requirements.txt
```

## File Structure
```
├── dynamic_prune.py
├── eval.py
├── nets
│   ├── alexnet.py
│   ├── resnet.py
│   └── vgg.py
├── pretrained_models
│   ├── resnet20_cifar10_original.pt
│   └── resnet50_cifar10_original.pt
├── pretrain_model.py
├── pruned_models
│   ├── resnet20_cifar10_updated_0.3.pt
│   ├── resnet20_cifar10_updated_0.5.pt
│   ├── resnet20_cifar10_updated_0.7.pt
│   └── resnet20_cifar10_updated_0.5.pt
├── README.md
├── requirements.txt
├── results
│   ├── resnet20_cifar10_original_results.png
│   ├── resnet20_cifar10_updated_0.3_results.png
│   ├── resnet20_cifar10_updated_0.5_results.png
│   └── resnet20_cifar10_updated_0.7_results.png
```

## Usage
1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Adjust hyperparameters and configurations as needed by modifying the script (pretrain_model.py). The hyperparameters are as follows: 
- choose the net you want from the **nets directory**
- learning_rate = 0.001 (default)
- momentum = 0.9 (default)
- epochs = 30 (default)
- depth (... of the resnet network) = 20 (default)
- dataset_name: cifar10 (default)

3. Train a model:
Run the desired Python script for training.
```bash
python pretrain_model.py
```

4. Adjust additional hyperparameters and configurations as needed by modifying the script (dynamic_prune.py). The hyperparameters are as follows: 
- choose the net you want from the **nets directory**
- learning_rate = 0.001 (default)
- momentum = 0.9 (default)
- epochs = 30 (default)
- depth (... of the resnet network) = 20 (default)
- dataset_name: cifar10 (default)
- beta (sparsity threshold) = 0.7 (default)

5. Perform Global Dynamic Pruning on the model along with its finetuning: Run the desired Python script for pruning and finetuning. 
```bash
python dynamic_prune.py
```

6. Evaluate trained models: After training, evaluate the trained and pruned models using evaluation scripts provided or custom evaluation code. This script will generate the graphs for validation accuracies and losses.
```bash
python eval.py
```

## Contributing
Feel free to contribute by submitting bug reports, feature requests, or pull requests. Your contributions are highly appreciated!

For any inquiries or feedback, please contact [jjoshi48@gatech.edu](jjoshi48@gatech.edu).

## Refrences
Shaohui Lin, Rongrong Ji, Yuchao Li, Yongjian Wu, Feiyue Huang, and Baochang Zhang. 2018. Accelerating convolutional networks via global & dynamic filter pruning. In Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI'18). AAAI Press, 2425–2432.
