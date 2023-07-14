from model3_init import *

model3.load_weights('data/model3.h5')

# Load the saved history object from a file
with open('data/history3.pkl', 'rb') as f:
    history3 = pickle.load(f)
