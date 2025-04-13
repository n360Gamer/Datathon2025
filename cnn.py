import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder


#train and test data directory
data_dir = "isic-2024-challenge/train-image"
test_data_dir = ""


#load the train and test data
dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))
# test_dataset = ImageFolder(test_data_dir,transforms.Compose([
#     transforms.Resize((150,150)),transforms.ToTensor()
# ]))



