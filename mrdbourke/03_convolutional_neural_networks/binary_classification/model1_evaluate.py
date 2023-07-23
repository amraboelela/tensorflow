from common import *

model1 = load_model('data/model1.keras')

# Load the saved history object from a file
with open('data/history1.pkl', 'rb') as f:
    history1 = pickle.load(f)

# Check out the layers in our model
print(model1.summary())
plot_curves(history1, 1)
