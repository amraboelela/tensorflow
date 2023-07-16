from model4_init import *

model4.load_weights('data/model4.h5')

# Load the saved history object from a file
with open('data/history4.pkl', 'rb') as f:
    history4 = pickle.load(f)

print(model4.summary())
plot_curves(history4, 4)
