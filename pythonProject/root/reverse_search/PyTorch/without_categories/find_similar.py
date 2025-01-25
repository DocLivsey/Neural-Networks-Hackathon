import torch
from torchvision import transforms
from PIL import Image
import pickle
from model import get_pretrained_model


# Преобразования изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка модели
model = get_pretrained_model()

# Загрузка признаков и путей к изображениям
with open('features.pkl', 'rb') as f:
    features, path_list = pickle.load(f)


# Функция для нахождения похожих изображений
def find_similar_images(query_image_path, features, path_list, model, transform, top_k=5):
    query_image = Image.open(query_image_path).convert("RGB")
    query_image = transform(query_image).unsqueeze(0)

    with torch.no_grad():
        query_features = model(query_image)
        query_features = query_features.view(query_features.size(0), -1)

    similarities = torch.mm(query_features, features.t())
    similarities = similarities.squeeze(0)

    _, indices = similarities.topk(top_k)
    similar_image_paths = [path_list[idx] for idx in indices]

    return similar_image_paths


# Пример использования
query_image_path = 'путь/к/запросу/изображения.jpg'
similar_images = find_similar_images(query_image_path, features, path_list, model, transform)

print("Похожие изображения:")
for img_path in similar_images:
    print(img_path)
