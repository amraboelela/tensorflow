from common import *
from model2 import *
import tensorflow as tf
import subprocess
import pickle
from tensorflow.keras.models import load_model

model.load_weights('data/model2.h5')

print(model.summary())

# Load the saved history object from a file
with open('data/history_all_classes_10_percent_fine_tune.pkl', 'rb') as f:
    history_all_classes_10_percent_fine_tune = pickle.load(f)

compare_historys(original_history=history_all_classes_10_percent,
                 new_history=history_all_classes_10_percent_fine_tune,
                 initial_epochs=5)
subprocess.run(['mv', 'plot.png', imagePath + "/plot2.png"])
