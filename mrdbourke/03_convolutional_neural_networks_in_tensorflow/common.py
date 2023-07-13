import sys
sys.path.append('../modules')

import numpy as np
import pandas as pd
import pickle
import random
import subprocess
import tensorflow as tf

from helper_functions import *
from os import path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the seed
tf.random.set_seed(42)

# Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Setup the train and test directories
train_dir = "data/pizza_steak/train/"
test_dir = "data/pizza_steak/test/"

# Import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32, # number of images to process at a time
                                               target_size=(224, 224), # convert all images to be 224 x 224
                                               class_mode="binary", # type of problem we're working on
                                               seed=42)

valid_data = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)
                                               
# Turn .csv files into pandas DataFrame's
#train_df = pd.read_csv("data/train.csv")
#test_df = pd.read_csv("data/test.csv")

# Shuffle training dataframe
#train_df_shuffled = train_df.sample(frac=1, random_state=42) # shuffle with random_state=42 for reproducibility

# Use train_test_split to split training data into training and validation sets
#train_sentences, val_sentences, train_labels, val_labels = train_test_split(
#        train_df_shuffled["text"].to_numpy(),
#        train_df_shuffled["target"].to_numpy(),
#        test_size=0.1, # dedicate 10% of samples to validation set
#        random_state=42 # random state for reproducibility
#    )
                                                                            
#imagePath = "/home/amr/webserver/HaneinWebserver/Public/images"
print("")
print("")

