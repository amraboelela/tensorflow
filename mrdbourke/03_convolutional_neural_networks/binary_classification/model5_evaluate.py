from model5_init import *

model5.load_weights('data/model5.keras')

# Load the saved history object from a file
with open('data/history5.pkl', 'rb') as f:
    history5 = pickle.load(f)

print(model5.summary())
plot_curves(history5, 5)
