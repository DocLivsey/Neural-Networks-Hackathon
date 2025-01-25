import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import load_data


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        print('Эпоха {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Обучение
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        print('Обучение: Потери {:.4f} Точность: {:.4f}'.format(
            epoch_loss, epoch_acc))

        # Валидация
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = running_corrects.double() / dataset_sizes['val']

        print('Валидация: Потери {:.4f} Точность: {:.4f}'.format(
            epoch_loss, epoch_acc))


def validate_model(model, criterion, val_loader):
    pass
    # Функция валидации модели


def save_model(model, filepath):
    model = models.resnet18(pretrained=True)

    # Заменяем последний слой (fully connected layer) на новый, который будет соответствовать количеству классов
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Выводим архитектуру модели
    print(model)
    # Функция сохранения модели


def load_model(filepath):
    pass
    # Функция загрузки модели
