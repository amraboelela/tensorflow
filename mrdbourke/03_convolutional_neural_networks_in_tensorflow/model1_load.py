from model1_init import *

model1.load_weights('data/model1.h5')

# Load the saved history object from a file
with open('data/history1.pkl', 'rb') as f:
    history1 = pickle.load(f)
