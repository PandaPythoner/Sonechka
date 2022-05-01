

from constants import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.utils import make_grid

def get_mnist_dataloader():
    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
    ])
    train_data = datasets.MNIST(
        root=input_images_path,
        train=True,
        download=True,
        transform=transform
    )
    train_data = torch.utils.data.Subset(train_data, list(range(0, 25000)))
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return dataloader


def get_ffhq_dataloader():
    transform = transforms.Compose([
        lambda x: x.resize((imgw, imgh)),
        transforms.ToTensor(),
        transforms.Normalize(
               mean=[0.5, 0.5, 0.5],
               std=[0.5, 0.5, 0.5]
        )                            
    ])
    train_data = datasets.ImageFolder(
        root=input_images_path,
        transform=transform
    )
    train_data = torch.utils.data.Subset(train_data, list(range(0, 20000)))
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return dataloader    

get_dataloader = get_ffhq_dataloader