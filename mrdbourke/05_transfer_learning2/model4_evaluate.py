from model4_init import *

model4.load_weights('data/model4.keras')

# Load the saved history object from a file
with open('data/history4.pkl', 'rb') as f:
    history4 = pickle.load(f)
  
print(model4.summary())

print("")
print("# Evaluate on the test data")
results_10_percent_data_aug = model4.evaluate(test_data)
print(results_10_percent_data_aug)

print("")
print("# Load in checkpoint saved model weights and evaluate model")
model4_evaluate = read_tensor("model4_evaluate")
if model4_evaluate is None:
    model4_evaluate = model4.evaluate(test_data)
    save_tensor(model4_evaluate, "model4_evaluate")
print(model4_evaluate)

print("")
print("# If the results from our native model and the loaded weights are the same, this should output True")
print(results_10_percent_data_aug == loaded_weights_model_results)

print("")
print("# Check to see if loaded model results are very close to native model results (should output True)")
print(np.isclose(np.array(results_10_percent_data_aug), np.array(loaded_weights_model_results)))

print("")
print("# Check the difference between the two results")
print(np.array(results_10_percent_data_aug) - np.array(loaded_weights_model_results))

plot_curves(history4, 4)

print("")
print("# Layers in loaded model")
print(model4.layers)

for layer in model4.layers:
  print(layer.trainable)
  
print("")
print("# How many layers are trainable in our base model?")
print(len(model4.layers[2].trainable_variables)) # layer at index 2 is the EfficientNetB0 layer (the base model)

print("")
print("print(len(base_model.trainable_variables))")

# Check which layers are tuneable (trainable)
for layer_number, layer in enumerate(base_model.layers):
  print(layer_number, layer.name, layer.trainable)

