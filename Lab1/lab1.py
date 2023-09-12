import torch
import torchvision
import torchvision.transforms as transforms

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.MNIST('./data/mnist', train=True, download=True, transform=train_transform)