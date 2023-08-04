from model1_init import *

model1.load_weights('data/model1.keras')

# Load the saved history object from a file
with open('data/history1.pkl', 'rb') as f:
    history1 = pickle.load(f)
    
# Check layers in our base model
for layer_number, layer in enumerate(base_model.layers):
  print(layer_number, layer.name)
  
print(base_model.summary())
  
print(model1.summary())
plot_curves(history1, 1)

# Run in terminal % tensorboard --logdir ./data/transfer_learning
