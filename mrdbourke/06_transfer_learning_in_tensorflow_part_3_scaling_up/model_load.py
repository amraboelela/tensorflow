from common import *
from model import *
import tensorflow as tf
import subprocess
import pickle
from tensorflow.keras.models import load_model

model.load_weights('data/model.h5')

print(model.summary())

# Load the saved history object from a file
with open('data/history_all_classes_10_percent.pkl', 'rb') as f:
    history_all_classes_10_percent = pickle.load(f)

plot_loss_curves(history_all_classes_10_percent)

subprocess.run(['mv', 'plot.png', imagePath + "/plot1.png"])

