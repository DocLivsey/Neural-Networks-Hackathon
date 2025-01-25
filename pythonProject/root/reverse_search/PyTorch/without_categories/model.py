import torch.nn as nn
from torchvision import models


def get_pretrained_model():
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()  # Переключение модели в режим оценки
    return model
