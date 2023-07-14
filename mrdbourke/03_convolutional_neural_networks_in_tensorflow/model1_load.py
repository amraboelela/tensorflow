from model1_init import *

model.load_weights('data/model1.h5')

# Load the saved history object from a file
with open('data/history_1.pkl', 'rb') as f:
    history_1 = pickle.load(f)

