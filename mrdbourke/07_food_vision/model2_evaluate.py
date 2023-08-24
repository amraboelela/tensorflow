from model2_init import *

# Get a summary of our downloaded model
print(model2.summary())

print("")
print("# Evaluate model")
model2_evaluate = read_tensor("model2_evaluate")
if model2_evaluate is None:
    model2_evaluate = model2.evaluate(test_data)
    save_tensor(model2_evaluate, "model2_evaluate")
print(model2_evaluate)

print("")
print("# Check the layers in the base model and see what dtype policy they're using")
for layer in model2.layers[1].layers[:20]:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)
