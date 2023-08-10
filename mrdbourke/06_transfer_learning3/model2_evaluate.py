from model2_init import *

print("")
print("# What layers in the model are trainable?")
for layer in model2.layers:
    print(layer.name, layer.trainable)

print("")
print("# Check which layers are trainable")
for layer_number, layer in enumerate(model2.layers[2].layers):
    print(layer_number, layer.name, layer.trainable)

print(model2.summary())

model2.load_weights(checkpoint_path(2))

# Evaluate fine-tuned model on the whole test dataset
model2_evaluate = read_tensor("model2_evaluate")
if model2_evaluate is None:
    model2_evaluate = model2.evaluate(test_data)
    save_tensor(model2_evaluate, "model2_evaluate")
print(model2_evaluate)

# Load the saved history object from a file
with open('data/history1.pkl', 'rb') as f:
    history1 = pickle.load(f)
    
# Load the saved history object from a file
with open('data/history2.pkl', 'rb') as f:
    history2 = pickle.load(f)
    
compare_historys(
    original_history=history1,
    new_history=history2,
    initial_epochs=5
)
                 
plot_curves(history2, 2)
