from model2_init import *

model.load_weights('data/model2.keras')

# Load the saved history object from a file
with open('data/history_all_classes_10_percent_fine_tune.pkl', 'rb') as f:
    history_all_classes_10_percent_fine_tune = pickle.load(f)

