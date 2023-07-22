from model10_init import *

model10 = load_model("data/model10.keras")

print("")
print("# Load the saved history object from a file")
with open('data/history10.pkl', 'rb') as f:
    history10 = pickle.load(f)

print("")
print("# Evaluate model on the test dataset")
print(model10.evaluate(X_test, y_test))

print("")
print("# Plot the decision boundaries for the training and test sets")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model10, X=X_train, y=y_train, index=10)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model10, X=X_test, y=y_test, index=10)

print("")
print("# Check the accuracy of our model")
loss, accuracy = model10.evaluate(X_test, y_test)
print(f"Model loss on test set: {loss}")
print(f"Model accuracy on test set: {(accuracy*100):.2f}%")

print("")
print("# Make predictions")
y_preds = model10.predict(X_test)

print("# Create confusion matrix")
#confusion_matrix(y_test, y_preds)

print("")
print("# View the first 10 predictions")
print(y_preds[:10])

print("")
print("# View the first 10 test labels")
print(y_test[:10])

print("")
print("# Convert prediction probabilities to binary format and view the first 10")
print(tf.round(y_preds)[:10])

print("")
print("# Create a confusion matrix")
print(confusion_matrix(y_test, tf.round(y_preds)))

# Note: The following confusion matrix code is a remix of Scikit-Learn's
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# and Made with ML's introductory notebook - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb

plt.figure()
figsize = (10, 10)

print("")
print("# Create the confusion matrix")
cm = confusion_matrix(y_test, tf.round(y_preds))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
n_classes = cm.shape[0]

# Let's prettify it
fig, ax = plt.subplots(figsize=figsize)
# Create a matrix plot
cax = ax.matshow(cm, cmap=plt.cm.Blues) # https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.axes.Axes.matshow.html
fig.colorbar(cax)

# Create classes
classes = False

if classes:
  labels = classes
else:
  labels = np.arange(cm.shape[0])

print("# Label the axes")
ax.set(title="Confusion Matrix",
       xlabel="Predicted label",
       ylabel="True label",
       xticks=np.arange(n_classes),
       yticks=np.arange(n_classes),
       xticklabels=labels,
       yticklabels=labels)

print("# Set x-axis labels to bottom")
ax.xaxis.set_label_position("bottom")
ax.xaxis.tick_bottom()

print("# Adjust label size")
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)
ax.title.set_size(20)

print("# Set threshold for different colors")
threshold = (cm.max() + cm.min()) / 2.

print("# Plot the text on each cell")
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
  plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
           horizontalalignment="center",
           color="white" if cm[i, j] > threshold else "black",
           size=15)
plt.savefig('data/images/confusion_matrix.png', format='png')
    
print("")
print("# What does itertools.product do? Combines two things into each combination")
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
  print(i, j)
