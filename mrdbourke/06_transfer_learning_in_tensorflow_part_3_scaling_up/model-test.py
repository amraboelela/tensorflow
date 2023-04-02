from common import *
from model import *
import tensorflow as tf
from tensorflow.keras.models import load_model

model.load_weights('data/model.h5')

print(model.summary())

# Evaluate model
#results_feature_extraction_model = model.evaluate(test_data)
#print("results_feature_extraction_model: " + str(results_feature_extraction_model))

# Load the history
#history_all_classes_10_percent = model.history.load('history_all_classes_10_percent.pkl')

import pickle

# Load the saved history object from a file
with open('data/history_all_classes_10_percent.pkl', 'rb') as f:
    history_all_classes_10_percent = pickle.load(f)

plot_loss_curves(history_all_classes_10_percent)
