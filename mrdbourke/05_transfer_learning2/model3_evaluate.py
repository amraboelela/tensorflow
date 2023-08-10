from model3_init import *

model3.load_weights('data/model3.keras')

# Load the saved history object from a file
with open('data/history3.pkl', 'rb') as f:
    history3 = pickle.load(f)
    
# Check layers in our base model
for layer_number, layer in enumerate(base_model.layers):
  print(layer_number, layer.name)
  
print(base_model.summary())
  
print(model3.summary())
plot_curves(history3, 3)
