from common import *

# Load the evaluation results from the file
with open('data/evaluation3.txt', 'r') as f:
    result_lines = f.readlines()

# Parse the evaluation results
loaded_loss = float(result_lines[0].split(': ')[1])
loaded_accuracy = float(result_lines[1].split(': ')[1])

pred_probs = np.loadtxt('data/predictions3.txt')

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

