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
import tensorflow_datasets as tfds

# List available datasets
datasets_list = tfds.list_builders() # get all available datasets in TFDS
print("food101" in datasets_list) # is the dataset we're after available?
