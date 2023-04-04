from helper_functions import *
import tensorflow as tf
from os import path
import subprocess
import pickle
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import random

train_dir = "data/101_food_classes_10_percent/train/"
test_dir = "data/101_food_classes_10_percent/test/"

# Setup data inputs
IMG_SIZE = (224, 224)
train_data_all_10_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                                label_mode="categorical",
                                                                                image_size=IMG_SIZE)

test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE,
                                                                shuffle=False) # don't shuffle test data for prediction analysis
class_names = test_data.class_names

imagePath = "/home/amr/webserver/HaneinWebserver/Public/images"
print("")
print("")

