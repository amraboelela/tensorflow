
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

# Turn .csv files into pandas DataFrame's
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Shuffle training dataframe
train_df_shuffled = train_df.sample(frac=1, random_state=42) # shuffle with random_state=42 for reproducibility

# Use train_test_split to split training data into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
        train_df_shuffled["target"].to_numpy(),
        test_size=0.1, # dedicate 10% of samples to validation set
        andom_state=42) # random state for reproducibility
                                                                            
imagePath = "/home/amr/webserver/HaneinWebserver/Public/images"
print("")
print("")

