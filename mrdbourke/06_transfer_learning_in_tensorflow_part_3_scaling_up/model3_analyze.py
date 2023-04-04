from model3_evaluate_load import *

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

fig, ax = plt.subplots(figsize=(12, 25))
scores = ax.barh(range(len(f1_scores)), f1_scores["f1-score"].values)
ax.set_yticks(range(len(f1_scores)))
ax.set_yticklabels(list(f1_scores["class_name"]))
ax.set_xlabel("f1-score")
ax.set_title("F1-Scores for 10 Different Classes")
ax.invert_yaxis(); # reverse the order

def autolabel(rects): # Modified version of: https://matplotlib.org/examples/api/barchart_demo.html
  """
  Attach a text label above each bar displaying its height (it's value).
  """
  for rect in rects:
    width = rect.get_width()
    ax.text(1.03*width, rect.get_y() + rect.get_height()/1.5,
            f"{width:.2f}",
            ha='center', va='bottom')
  
autolabel(scores)
plt.savefig('plot.png', format='png')
subprocess.run(['mv', 'plot.png', imagePath + "/plot3.png"])

