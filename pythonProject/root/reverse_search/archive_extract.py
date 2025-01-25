import zipfile


zip_archive = zipfile.ZipFile('../Resources/training.zip', 'r')
zip_archive.extractall('../Resources')
zip_archive.close()
