from common1 import *

print(tf.__version__) # check the version (should be 2.x+)

print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

print("")
print("# Visualize it")
plt.scatter(X, y)
plt.savefig('data/images/scatter.png', format='png')

print("")
print("# Example input and output shapes of a regression model")
house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant([939700])
print(house_info, house_price)
print(house_info.shape)

print("")
print("# Take a single example of X")
input_shape = X[0].shape

print("# Take a single example of y")
output_shape = y[0].shape
print(input_shape, output_shape, "# these are both scalars (no shape)")

print("")
print("# Let's take a look at the single examples invidually")
print(X[0], y[0])

print("")
print("# Make a bigger dataset")
X = np.arange(-100, 100, 4)
print(X)

print("")
print("# Make labels for the dataset (adhering to the same pattern as before)")
y = np.arange(-90, 110, 4)
print(y)

print("")
print("# Same result as above")
y = X + 10
print(y)

print("")
print("# Check how many samples we have")
print(len(X))

print("")
print("# Split data into train and test sets")
X_train = X[:40] # first 40 examples (80% of data)
y_train = y[:40]

X_test = X[40:] # last 10 examples (20% of data)
y_test = y[40:]

print(len(X_train), len(X_test))

plt.figure(figsize=(10, 7))
# Plot training data in blue
plt.scatter(X_train, y_train, c='b', label='Training data')
# Plot test data in green
plt.scatter(X_test, y_test, c='g', label='Testing data')
# Show the legend
plt.legend()
plt.savefig('data/images/train_test.png', format='png')

print("")
print("# Read in the insurance dataset")
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

print("# Check out the insurance dataset")
print(insurance.head())
