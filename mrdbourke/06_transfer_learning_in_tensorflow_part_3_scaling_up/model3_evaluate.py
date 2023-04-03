from model3_load import *
from confusion_matrix import *

# Check to see if loaded model is a trained model
#loaded_loss, loaded_accuracy = model.evaluate(test_data)
#print(loaded_loss)
#print(loaded_accuracy)

# Make predictions with model
pred_probs = model.predict(test_data, verbose=1) # set verbosity to see how long it will take
print(len(pred_probs))
print(pred_probs.shape)
print(pred_probs[:10])

# We get one prediction probability per class
print(f"Number of prediction probabilities for sample 0: {len(pred_probs[0])}")
print(f"What prediction probability sample 0 looks like:\n {pred_probs[0]}")
print(f"The class with the highest predicted probability by the model for sample 0: {pred_probs[0].argmax()}")

# Get the class predicitons of each label
pred_classes = pred_probs.argmax(axis=1)

# How do they look?
print(pred_classes[:10])

# Note: This might take a minute or so due to unravelling 790 batches
y_labels = []
for images, labels in test_data.unbatch(): # unbatch the test data and get images and labels
  y_labels.append(labels.numpy().argmax()) # append the index which has the largest value (labels are one-hot)
print(y_labels[:10]) # check what they look like (unshuffled)

# How many labels are there? (should be the same as how many prediction probabilities we have)
len(y_labels)

# Get accuracy score by comparing predicted classes to ground truth labels
#from sklearn.metrics import accuracy_score
#sklearn_accuracy = accuracy_score(y_labels, pred_classes)
#print("sklearn_accuracy: " + str(sklearn_accuracy))

# Does the evaluate method compare to the Scikit-Learn measured accuracy?
#import numpy as np
#print(f"Close? {np.isclose(loaded_accuracy, sklearn_accuracy)} | Difference: {loaded_accuracy - sklearn_accuracy}")

# Get the class names
class_names = test_data.class_names
print(class_names)

# Plot a confusion matrix with all 25250 predictions, ground truth labels and 101 classes
make_confusion_matrix(y_true=y_labels,
                      y_pred=pred_classes,
                      classes=class_names,
                      figsize=(100, 100),
                      text_size=20,
                      norm=False,
                      savefig=True)

subprocess.run(['mv', 'confusion_matrix.png', imagePath])