import re
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from time import time

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

dataset_path = "../Resources/datasets/train"
