from model3_init import *

# Check to see if loaded model is a trained model
#loaded_loss, loaded_accuracy = model3.evaluate(test_data)
#print(loaded_loss, loaded_accuracy)

# Make predictions with model
pred_probs = model3.predict(test_data, verbose=1) # set verbosity to see how long it will take

print("")
print("# How many predictions are there?")
print(len(pred_probs))

print("")
print("# What's the shape of our predictions?")
print(pred_probs.shape)


print("")
print("# How do they look?")
print(pred_probs[:10])

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
