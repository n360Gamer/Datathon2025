import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# !pip install torchvision
import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# !pip install torchmetrics
import torchmetrics

class CNN(nn.Module):
   def __init__(self, in_channels, num_classes):

       """
       Building blocks of convolutional neural network.

       Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
       """
       super(CNN, self).__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
       # Max pooling layer
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       # 2nd convolutional layer
       self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       # Fully connected layer
       self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

   def forward(self, x):
       """
       Define the forward pass of the neural network.

       Parameters:
           x: Input tensor.

       Returns:
           torch.Tensor
               The output tensor after passing through the network.
       """
       x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = x.reshape(x.shape[0], -1)  # Flatten the tensor
       x = self.fc1(x)            # Apply fully connected layer
       return x
       x = x.reshape(x.shape[0], -1)  # Flatten the tensor
       x = self.fc1(x)            # Apply fully connected layer
       return x



def main():
    batch_size = 60

    train_dataset = datasets.DatasetFolder(root = "train-image/image/",
                                           transform = transforms.Compose([transforms.Resize((150,150)),transforms.ToTensor()]))

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

    test_dataset = datasets.MNIST(root = "train-image/image/", download = True, train = False,
                                  transform = transforms.ToTensor())

    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNN(in_channels = 1, num_classes = 10).to(device)
    print(model)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    num_epochs=10
    for epoch in range(num_epochs):
     # Iterate over training batches
       print(f"Epoch [{epoch + 1}/{num_epochs}]")

       for batch_index, (data, targets) in enumerate(tqdm(dataloader_train)):
           data = data.to(device)
           targets = targets.to(device)
           scores = model(data)
           loss = criterion(scores, targets)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()