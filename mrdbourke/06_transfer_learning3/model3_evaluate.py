from model3_init import *

print("")
print("# Check to see if loaded model is a trained model")
model3_evaluate = read_tensor("model3_evaluate")
if model3_evaluate is None:
    model3_evaluate = model3.evaluate(test_data)
    save_tensor(model3_evaluate, "model3_evaluate")
print(model3_evaluate)
model3_loss, model3_accuracy = model3_evaluate

print("")
print("# Load the saved history object from a file")
with open('data/history3.pkl', 'rb') as f:
    history3 = pickle.load(f)
 
plot_curves(history3, 3)

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
print(f"The class with the highest predicted probability by the model for sample 0: {tf.argmax(pred_probs[0])}")

print("")
print("# Get the class predicitons of each label")
pred_classes = tf.argmax(pred_probs, axis=1)

print("")
print("# How do they look?")
print(pred_classes[:10])

print("")
print("# Note: This might take a minute or so due to unravelling 790 batches")
y_labels = []
for images, labels in test_data.unbatch(): # unbatch the test data and get images and labels
    y_labels.append(labels.numpy().argmax()) # append the index which has the largest value (labels are one-hot)
print(y_labels[:10], "# check what they look like (unshuffled)")

print("")
print("# How many labels are there? (should be the same as how many prediction probabilities we have)")
print(len(y_labels))

print("")
print("# Get accuracy score by comparing predicted classes to ground truth labels")
sklearn_accuracy = accuracy_score(y_labels, pred_classes)
print(sklearn_accuracy)

print("")
print("# Does the evaluate method compare to the Scikit-Learn measured accuracy?")
print(f"Close? {np.isclose(model3_accuracy, sklearn_accuracy)} | Difference: {model3_accuracy - sklearn_accuracy}")

print("")
print("# Get the class names")
class_names = test_data.class_names
print(class_names[:10])

# Plot a confusion matrix with all 25250 predictions, ground truth labels and 101 classes
make_confusion_matrix(
    y_true=y_labels,
    y_pred=pred_classes,
    classes=class_names,
    figsize=(100, 100),
    text_size=20,
    norm=False
)

print(classification_report(y_labels, pred_classes))

print("")
print("# Get a dictionary of the classification report")
classification_report_dict = classification_report(y_labels, pred_classes, output_dict=True)
print(classification_report_dict)

print("")
# Create empty dictionary
class_f1_scores = {}
print("# Loop through classification report items")
for k, v in classification_report_dict.items():
  if k == "accuracy": # stop once we get to accuracy key
    break
  else:
    # Append class names and f1-scores to new dictionary
    class_f1_scores[class_names[int(k)]] = v["f1-score"]
print(class_f1_scores)

print("")
print("# Turn f1-scores into dataframe for visualization")
f1_scores = pd.DataFrame({"class_name": list(class_f1_scores.keys()),
                          "f1-score": list(class_f1_scores.values())}).sort_values("f1-score", ascending=False)
print(f1_scores.head())

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
with open('data/history2.pkl', 'rb') as f:
    history2 = pickle.load(f)

compare_historys(
    original_history=history1,
    new_history=history2,
    initial_epochs=5
)
                 
plot_curves(history2, 2)
