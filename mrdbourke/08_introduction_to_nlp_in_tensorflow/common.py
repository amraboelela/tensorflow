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

# Turn .csv files into pandas DataFrame's
import pandas as pd
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
print(str(train_df.head()))

imagePath = "/home/amr/webserver/HaneinWebserver/Public/images"
print("")
print("")

