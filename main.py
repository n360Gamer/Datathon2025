# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset

import os
import pandas as pd


# Creating a CNN class
class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(18496, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class SkinLesionDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, low_memory=False)

        self.img_labels = self.img_labels[['isic_id', 'target']]
        # Filter out rows where the image file does not exist
        self.img_labels = self.img_labels[self.img_labels['isic_id'].apply(
            lambda x: os.path.exists(os.path.join(img_dir, x + '.jpg')))]
        self.img_labels = self.img_labels.reset_index(drop=True)

        if self.img_labels.empty:
            raise ValueError(
                "No valid image files found in the specified directory.")

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, self.img_labels.iloc[idx, 0] + '.jpg')
        image = read_image(img_path).float() / 255.0
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Define relevant variables for the ML task
num_classes = 2
learning_rate = 0.01
num_epochs = 2

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use
all_transforms = transforms.Compose([transforms.Resize((80, 80))])
# Create Training dataset
dp = SkinLesionDataset('isic-2024-challenge/train-metadata.csv', 'isic-2024-challenge/train-image/image',
                       transform=all_transforms)

# TODO: Add labels to data before splitting.

train_size, test_size = int(len(dp) * 0.8), len(dp) - (int(len(dp) * 0.8))
train_dataset, test_dataset = torch.utils.data.random_split(
    dp, [train_size, test_size])

# create batch sizes for train and test dataloaders
# (loading everything into memory, no minibatches)
# batch_train, batch_test = len(train_dataset), len(test_dataset)
batch_train, batch_test = (64,64)
# create train and test dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_train, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_test)


model = ConvNeuralNet(num_classes)

# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# Set optimizer with optimizer
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

total_step = len(train_dataloader)

# We use the pre-defined number of epochs to determine how many iterations to train the network on
for epoch in range(num_epochs):
    # Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(train_dataloader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
          1, num_epochs, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_dataloader:
        images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} train images: {} %'.format(
        40000, 100 * correct / total))
