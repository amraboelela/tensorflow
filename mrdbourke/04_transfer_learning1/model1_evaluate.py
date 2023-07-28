from model1_load import *

model1.load_weights('data/model1.keras')

# Load the saved history object from a file
with open('data/model1.pkl', 'rb') as f:
    model1 = pickle.load(f)
    
print(model1.summary())
plot_loss_curves(history1)

