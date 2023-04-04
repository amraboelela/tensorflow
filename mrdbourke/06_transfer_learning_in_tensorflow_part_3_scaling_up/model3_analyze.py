from model3_evaluate_load import *
from confusion_matrix import *

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
print(len(y_labels))

# Get accuracy score by comparing predicted classes to ground truth labels
sklearn_accuracy = accuracy_score(y_labels, pred_classes)
print("sklearn_accuracy: " + str(sklearn_accuracy))

# Does the evaluate method compare to the Scikit-Learn measured accuracy?
print(f"Close? {np.isclose(loaded_accuracy, sklearn_accuracy)} | Difference: {loaded_accuracy - sklearn_accuracy}")

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

from sklearn.metrics import classification_report
print(classification_report(y_labels, pred_classes))

classification_report_dict = classification_report(y_labels, pred_classes, output_dict=True)
print(classification_report_dict)

# Create empty dictionary
class_f1_scores = {}
# Loop through classification report items
for k, v in classification_report_dict.items():
  if k == "accuracy": # stop once we get to accuracy key
    break
  else:
    # Append class names and f1-scores to new dictionary
    class_f1_scores[class_names[int(k)]] = v["f1-score"]
print(class_f1_scores)

# Turn f1-scores into dataframe for visualization
f1_scores = pd.DataFrame({"class_name": list(class_f1_scores.keys()),
                          "f1-score": list(class_f1_scores.values())}).sort_values("f1-score", ascending=False)
print(f1_scores)

