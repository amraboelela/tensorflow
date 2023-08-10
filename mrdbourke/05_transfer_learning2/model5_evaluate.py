from model5_init import *

model5.load_weights(checkpoint_path(5))

# Load the saved history object from a file
with open('data/history4.pkl', 'rb') as f:
    history4 = pickle.load(f)
    
# Load the saved history object from a file
with open('data/history5.pkl', 'rb') as f:
    history5 = pickle.load(f)
    
print("")
print("# Check which layers are tuneable (trainable)")
for layer_number, layer in enumerate(model5.layers[2].layers):
  print(layer_number, layer.name, layer.trainable)
  
print(len(model5.trainable_variables))
  
print(model5.summary())
plot_curves(history5, 5)

print("")
print("# Evaluate the model on the test data")
model5_evaluate = read_tensor("model5_evaluate")
if model5_evaluate is None:
    model5_evaluate = model5.evaluate(test_data)
    save_tensor(model5_evaluate, "model5_evaluate")
print(model5_evaluate)

compare_historys(
    original_history=history4,
    new_history=history5,
    initial_epochs=5
)

