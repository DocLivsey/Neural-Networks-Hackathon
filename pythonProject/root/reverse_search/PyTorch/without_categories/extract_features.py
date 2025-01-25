import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pickle
from dataset import CustomImageDataset
from model import get_pretrained_model


# Преобразования изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Путь к папке с изображениями
image_dir = '/content/drive/MyDrive/training'

# Создание датасета
dataset = CustomImageDataset(image_dir=image_dir, transform=transform)

# Создание загрузчика данных
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Получение предобученной модели
model = get_pretrained_model()

# Извлечение признаков
feature_list = []
path_list = []

with torch.no_grad():
    for inputs, paths in data_loader:
        outputs = model(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        feature_list.append(outputs)
        path_list.extend(paths)

# Объединение всех признаков в один тензор
features = torch.cat(feature_list)

# Сохранение признаков и путей к изображениям
with open('features.pkl', 'wb') as f:
    pickle.dump((features, path_list), f)

print("Извлечение признаков завершено и сохранено в 'features.pkl'")
