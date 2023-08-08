from model2_init import *

print("")
print("# What layers in the model are trainable?")
for layer in model2.layers:
    print(layer.name, layer.trainable)

print("")
print("# Check which layers are trainable")
for layer_number, layer in enumerate(model2.layers[2].layers):
    print(layer_number, layer.name, layer.trainable)
  
exit()

print(model2.summary())

model2.load_weights(checkpoint_path(2))

# Load the saved history object from a file
with open('data/history2.pkl', 'rb') as f:
    history2 = pickle.load(f)

print("")
print("# Evaluate model")
results_feature_extraction_model = model1.evaluate(test_data)
print(results_feature_extraction_model)

plot_curves(history2, 2)

# Run in terminal % tensorboard --logdir ./data/transfer_learning
