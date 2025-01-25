import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

def load_data(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    return dataset
