

from constants import *


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import imageio
import numpy as np
import matplotlib
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = imgsz
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = x.view(-1, imgsz)
        return self.main(x)
    
class Generator(nn.Module):
    def __init__(self, generator_input_sz):
        super(Generator, self).__init__()
        self.generator_input_sz = generator_input_sz
        self.main = nn.Sequential(
            nn.Linear(self.generator_input_sz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, imgw * imgh * 3),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.main(x).view(-1, 3, imgw, imgh)
    
def create_noise(sample_size, generator_input_sz, device):
    return torch.randn(sample_size, generator_input_sz).to(device)