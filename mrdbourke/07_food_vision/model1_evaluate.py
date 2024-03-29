from model1_init import *

print(model1.summary())

# Check the dtype_policy attributes of layers in our model
for layer in model1.layers:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy) # Check the dtype policy of layers

# Check the layers in the base model and see what dtype policy they're using
for layer in model1.layers[1].layers[:20]: # only check the first 20 layers to save output space
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

model1.load_weights(checkpoint_path(1))

print("")
print("# Evaluate model")
model1_evaluate = read_tensor("model1_evaluate")
if model1_evaluate is None:
    model1_evaluate = model1.evaluate(test_data)
    save_tensor(model1_evaluate, "model1_evaluate")
print(model1_evaluate)

# Load the saved history object from a file
with open('data/history1.pkl', 'rb') as f:
    history1 = pickle.load(f)

plot_curves(history1, 1)
