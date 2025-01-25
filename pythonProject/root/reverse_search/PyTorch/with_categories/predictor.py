import torch
from torchvision import transforms
from PIL import Image
from model_trainer import load_model


def load_and_preprocess_image(image_path):
    # Обучение модели
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Разбиваем датасет на тренировочную и валидационную выборки
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [4000, 500])

    # Создаем DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Запускаем обучение модели
    train_model(model, criterion, optimizer, num_epochs=5)
    # Функция загрузки и предобработки изображения


def predict_image_class(image_path, model, class_names):
    pass
    # Функция предсказания класса изображения
