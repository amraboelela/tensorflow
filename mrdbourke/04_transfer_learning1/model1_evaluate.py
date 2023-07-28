from model1_init import *

model1.load_weights('data/model1.keras')

# Load the saved history object from a file
with open('data/history1.pkl', 'rb') as f:
    history1 = pickle.load(f)
    
print(model1.summary())
plot_loss_curves(history1)

