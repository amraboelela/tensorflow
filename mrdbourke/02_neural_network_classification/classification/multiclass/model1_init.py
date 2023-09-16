from common import *

# Set random seed
tf.random.set_seed(42)

# Create the model
model1 = Sequential([
    Flatten(input_shape=(28, 28)), # input layer (we had to reshape 28x28 to 784, the Flatten layer does this for us)
    Dense(4, activation="relu"),
    Dense(4, activation="relu"),
    Dense(10, activation="softmax") # output shape is 10, activation is softmax
])

# Compile the model
model1.compile(
    loss=SparseCategoricalCrossentropy(), # different loss function for multiclass classifcation
    optimizer=Adam(),
    metrics=["accuracy"]
)
