import sys
sys.path.append('../../modules')

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
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Make the creating of our model a little easier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
import pathlib

download("https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip")
