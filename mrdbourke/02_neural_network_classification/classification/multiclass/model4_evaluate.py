from model4_init import *

model4 = load_model("data/model4.keras")

print("")
print("# Load the saved history object from a file")
with open('data/history4.pkl', 'rb') as f:
    history4 = pickle.load(f)

print("")
print("# Make predictions with the most recent model")
y_probs = model4.predict(test_data) # "probs" is short for probabilities

print("# View the first 5 predictions")
print(y_probs[:5])

print("")
print("# See the predicted class number and label for the first example")
print(y_probs[0].argmax(), class_names[y_probs[0].argmax()])

print("")
print("# Convert all of the predictions from probabilities to labels")
y_preds = y_probs.argmax(axis=1)

print("# View the first 10 prediction labels")
print(y_preds[:10])

print("")
print("# Check out the non-prettified confusion matrix")
print(confusion_matrix(y_true=test_labels,
                       y_pred=y_preds))

print("")
print("# Make a prettier confusion matrix")
make_confusion_matrix(y_true=test_labels,
                      y_pred=y_preds,
                      classes=class_names,
                      figsize=(15, 15),
                      text_size=10)
                      
print("")
print("# Check out a random image as well as its prediction")
plot_random_image(model=model4,
                  images=test_data,
                  true_labels=test_labels,
                  classes=class_names)

print("")
print("# Find the layers of our most recent model")
print(model4.layers)

print("")
print("# Extract a particular layer")
print(model4.layers[1])

print("")
print("# Get the patterns of a layer in our network")
weights, biases = model4.layers[1].get_weights()

print("# Shape = 1 weight matrix the size of our input data (28x28) per neuron (4)")
print(weights, weights.shape)

print("")
print("# Shape = 1 bias per neuron (we use 4 neurons in the first layer)")
print(biases, biases.shape)

print("")
print("# Can now calculate the number of paramters in our model")
print(model4.summary())

print("")
print("# See the inputs and outputs of each layer")
plot_model(model4, to_file='data/images/model4.png', show_shapes=True)
