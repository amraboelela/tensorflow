from common import *

print(tf.__version__)
print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

print("# Check out the features")
print(X)

print("")
print("# See the first 10 labels")
print(y[:10])

print("")
print("# Make dataframe of features and labels")
circles = pd.DataFrame({"X0":X[:, 0], "X1":X[:, 1], "label":y})
print(circles.head())

print("")
print("# Check out the different labels")
print(circles.label.value_counts())

print("")
print("# Visualize with a plot")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu);
plt.savefig('data/images/circles.png', format='png')

print("")
print("# Check the shapes of our features and labels")
print(X.shape, y.shape)

print("")
print("# Check how many samples we have")
print(len(X), len(y))

print("")
print("# View the first example of features and labels")
print(X[0], y[0])

print("")
print("# Create a toy tensor (similar to the data we pass into our model)")
A = tf.cast(tf.range(-10, 10), tf.float32)
print(A)

print("")
print("# Visualize our toy tensor")
plt.figure()
plt.plot(A)
plt.savefig('data/images/A.png', format='png')

print("")
print("# Sigmoid - https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid")
print("# Use the sigmoid function on our tensor")
print(sigmoid(A))

print("")
print("# Plot sigmoid modified tensor")
plt.figure()
plt.plot(sigmoid(A))
plt.savefig('data/images/sigmoid_A.png', format='png')

print("")
print("# ReLU - https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu")

print("# Pass toy tensor through ReLU function")
print(relu(A))

print("")
print("# Plot ReLU-modified tensor")
plt.figure()
plt.plot(relu(A))
plt.savefig('data/images/relu_A.png', format='png')

print("")
print("# Linear - https://www.tensorflow.org/api_docs/python/tf/keras/activations/linear (returns input non-modified...)")
print(tf.keras.activations.linear(A))

print("")
print("# Does the linear activation change anything?")
print(A == tf.keras.activations.linear(A))

print("")
print("# How many examples are in the whole dataset?")
print(len(X))

print("")
print("# Check the shapes of the data")
print(X_train.shape, X_test.shape) # 800 examples in the training set, 200 examples in the test set

