from matplotlib import pyplot as plt
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from IPython.display import display
from PIL import Image
import shutil
import random
import os

gauth = GoogleAuth()
gauth.LocalWebserverAuth()


def create_and_upload_files(file_name, file_content):
    try:
        drive = GoogleDrive(gauth)

        file = drive.CreateFile({'title': file_name})
        file.SetContentFile(file_content)
        file.Upload()

        return f"file {file_name} was uploaded"
    except Exception as e:
        return str(e)


def read_all_files():
    try:
        drive = GoogleDrive(gauth)
        file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        for file in file_list:
            print(file)
    except Exception as e:
        return str(e)


def read_files_from_drive_dataset():
    drive = GoogleDrive(gauth)
    try:
        file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        dataset_id = ''
        for file in file_list:
            if file['title'] == 'unlabeled_dataset':
                dataset_id = file['id']

        categories = drive.ListFile(
            {'q': f"'{dataset_id}' in parents and mimeType = 'application/vnd.google-apps.folder'"}).GetList()
        for category in categories:
            pass

    except Exception as e:
        return str(e)


def read_files_from(path_to_folder='root'):
    drive = GoogleDrive(gauth)
    try:
        file_list = drive.ListFile({'q': f"title='{path_to_folder}'"}).GetList()
        for file in file_list:
            print(file)
    except Exception as e:
        return str(e)


def create_folder(new_folder_name, parent_folder_name):
    try:
        drive = GoogleDrive(gauth)
        file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        for file in file_list:
            if file['title'] == parent_folder_name:
                parent_folder_id = file['id']
                file_metadata = {
                    'title': new_folder_name,
                    'parents': [parent_folder_id],
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = drive.CreateFile(file_metadata)
                folder.Upload()
    except Exception as e:
        return str(e)


def create_labeled_training_dataset():
    drive = GoogleDrive(gauth)
    dataset_folder_id = ''

    file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file in file_list:
        if file['title'] == 'dataset':
            dataset_folder_id = file['id']

    training_folder_name = 'training'
    training_folder = drive.CreateFile(
        {'title': training_folder_name, 'mimeType': 'application/vnd.google-apps.folder'})
    training_folder.Upload()
    training_folder_id = training_folder['id']

    temp_folder = 'temp_images'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    file_list = drive.ListFile(
        {'q': f"'{dataset_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder'"}).GetList()
    for category_folder in file_list:
        category_name = category_folder['title']
        category_id = category_folder['id']

        training_category_folder = drive.CreateFile({
            'title': category_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [{'id': training_folder_id}]
        })
        training_category_folder.Upload()
        training_category_folder_id = training_category_folder['id']

        images = drive.ListFile({'q': f"'{category_id}' in parents and mimeType contains 'image/jpeg'"}).GetList()
        random.shuffle(images)
        selected_images = images[:100]

        for image in selected_images:
            # Загрузка изображения на локальный диск
            local_image_path = os.path.join(temp_folder, image['title'])
            image.GetContentFile(local_image_path)

            # Отображение изображения
            with Image.open(local_image_path) as img:
                display(img)

            # Загрузка изображения обратно на Google Drive в папку training
            copied_file = drive.CreateFile({
                'title': image['title'],
                'parents': [{'id': training_category_folder_id}]
            })
            copied_file.SetContentFile(local_image_path)
            copied_file.Upload()

    # Очистка временной папки
    shutil.rmtree(temp_folder)

    print("Файлы успешно скопированы!")


def create_unlabeled_dataset():
    drive = GoogleDrive(gauth)
    dataset_folder_id = ''

    file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file in file_list:
        if file['title'] == 'dataset':
            dataset_folder_id = file['id']

    unlabeled_folder_name = 'unlabeled_dataset'
    training_folder = drive.CreateFile(
        {'title': unlabeled_folder_name, 'mimeType': 'application/vnd.google-apps.folder'})
    training_folder.Upload()

    temp_folder = 'temp_images'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    file_list = drive.ListFile(
        {'q': f"'{dataset_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder'"}).GetList()
    for category_folder in file_list:
        category_id = category_folder['id']

        images = drive.ListFile({'q': f"'{category_id}' in parents and mimeType contains 'image/jpeg'"}).GetList()

        for image in images:
            local_image_path = os.path.join(temp_folder, image['title'])
            image.GetContentFile(local_image_path)

            with Image.open(local_image_path) as img:
                plt.imshow(img)

            copied_file = drive.CreateFile({
                'title': image['title'],
                'parents': [{'id': dataset_folder_id}]
            })
            copied_file.SetContentFile(local_image_path)
            copied_file.Upload()

    shutil.rmtree(temp_folder)

    print("Файлы успешно скопированы!")
