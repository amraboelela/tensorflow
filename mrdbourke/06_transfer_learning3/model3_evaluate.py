from model3_init import *

print("")
print("# Check to see if loaded model is a trained model")
model3_evaluate = read_tensor("model3_evaluate")
if model3_evaluate is None:
    model3_evaluate = model3.evaluate(test_data)
    save_tensor(model3_evaluate, "model3_evaluate")
print(model3_evaluate)

print("")
print("# Make predictions with model")
pred_probs = read_tensor("pred_probs")
if pred_probs is None:
    pred_probs = model3.predict(test_data, verbose=1) # set verbosity to see how long it will take
    save_tensor(pred_probs, "pred_probs")

print("")
print("# How many predictions are there?")
print(len(pred_probs))

print("")
print("# What's the shape of our predictions?")
print(pred_probs.shape)


print("")
print("# How do they look?")
print(pred_probs[:10])

print("")
print("# We get one prediction probability per class")
print(f"Number of prediction probabilities for sample 0: {len(pred_probs[0])}")
print(f"What prediction probability sample 0 looks like:\n {pred_probs[0]}")
print(f"The class with the highest predicted probability by the model for sample 0: {pred_probs[0].argmax()}")

exit()

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

# Load the saved history object from a file
with open('data/history1.pkl', 'rb') as f:
    history1 = pickle.load(f)
    
# Load the saved history object from a file
with open('data/history2.pkl', 'rb') as f:
    history2 = pickle.load(f)

# Evaluate fine-tuned model on the whole test dataset
#results_all_classes_10_percent_fine_tune = model2.evaluate(test_data)
#print(results_all_classes_10_percent_fine_tune)

compare_historys(
    original_history=history1,
    new_history=history2,
    initial_epochs=5
)
                 
plot_curves(history2, 2)

# Run in terminal % tensorboard --logdir ./data/transfer_learning
