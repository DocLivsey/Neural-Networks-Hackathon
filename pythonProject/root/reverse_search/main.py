from root.reverse_search.PyDrive.google_drive_agent import *

FOLDERS = [
    'dataset',
    'unlabeled_dataset',
    'training',
    'unlabeled_training',
]

if __name__ == '__main__':
    read_files_from_drive_dataset()
