from model1_init import *

model.load_weights('data/model.h5')

# Load the saved history object from a file
with open('data/history_all_classes_10_percent.pkl', 'rb') as f:
    history_all_classes_10_percent = pickle.load(f)

