from model7_init import *

model7.load_weights('data/model7.keras')

# Load the saved history object from a file
with open('data/history7.pkl', 'rb') as f:
    history7 = pickle.load(f)

print(model7.summary())
plot_curves(history7, 7)
