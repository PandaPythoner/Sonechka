

from constants import *
from dataloader import get_dataloader
from networks import *


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


matplotlib.style.use('ggplot')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


generator = Generator(generator_input_sz).to(device)
discriminator = Discriminator().to(device)
print('##### GENERATOR #####')
print(generator)
print('######################')
print('\n##### DISCRIMINATOR #####')
print(discriminator)
print('######################')


dataloader = get_dataloader()

optim_generator = optim.Adam(generator.parameters(), lr=0.0002)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002)

criterion = nn.BCELoss()


losses_g = [] # to store generator loss after each epoch
losses_d = [] # to store discriminator loss after each epoch
images = [] # to store images generatd by the generator


# to create real labels (1s)
def label_real(size):
    data = torch.ones(size, 1)
    return data.to(device)

# to create fake labels (0s)
def label_fake(size):
    data = torch.zeros(size, 1)
    return data.to(device)


def create_noise(sample_size, generator_input_sz):
    return torch.randn(sample_size, generator_input_sz).to(device)


def save_generator_image(image, path):
    save_image(image, path)


def train_discriminator(optimizer, data_real, data_fake):
    b_size = data_real.size(0)
    real_label = label_real(b_size)
    fake_label = label_fake(b_size)
    optimizer.zero_grad()
    output_real = discriminator(data_real)
    loss_real = criterion(output_real, real_label)
    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, fake_label)
    loss_real.backward()
    loss_fake.backward()
    optimizer.step()
    return loss_real + loss_fake


def train_generator(optimizer, data_fake):
    b_size = data_fake.size(0)
    real_label = label_real(b_size)
    optimizer.zero_grad()
    output = discriminator(data_fake)
    loss = criterion(output, real_label)
    loss.backward()
    optimizer.step()
    return loss


noise = create_noise(sample_size, generator_input_sz)


generator.train()
discriminator.train()


for epoch in range(epochs):
    loss_generator = 0.0
    loss_discriminator = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(dataloader)/dataloader.batch_size)):
        image, _ = data
        image = image.to(device)
        b_size = len(image)
        # run the discriminator for k number of steps
        for step in range(discriminator_train_steps):
            data_fake = generator(create_noise(b_size, generator_input_sz)).detach()
            data_real = image
            # train the discriminator network
            loss_discriminator += train_discriminator(optim_discriminator, data_real, data_fake)
        data_fake = generator(create_noise(b_size, generator_input_sz))
        # train the generator network
        loss_generator += train_generator(optim_generator, data_fake)
    # create the final fake image for the epoch
    generated_img = generator(noise).cpu().detach()
    # make the images as grid
    generated_img = make_grid(generated_img)
    # save the generated torch tensor models to disk
    save_generator_image(generated_img, f"./imgs/MNIST_generated/gen_img{epoch}.png")
    images.append(generated_img)
    epoch_loss_generator = loss_generator / bi # total generator loss for the epoch
    epoch_loss_discriminator = loss_discriminator / bi # total discriminator loss for the epoch
    losses_g.append(epoch_loss_generator)
    losses_d.append(epoch_loss_discriminator)
    
    print(f"Epoch {epoch} of {epochs}")
    print(f"Generator loss: {epoch_loss_generator:.8f}, Discriminator loss: {epoch_loss_discriminator:.8f}")