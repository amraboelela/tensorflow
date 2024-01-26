from common import *

# Divide train and test images by the maximum value (normalize it)
train_data = train_data / 255.0
test_data = test_data / 255.0

# Create the model
model3 = Sequential([
    Flatten(input_shape=(28, 28)), # input layer (we had to reshape 28x28 to 784)
    Dense(4, activation="relu"),
    Dense(4, activation="relu"),
    Dense(10, activation="softmax") # output shape is 10, activation is softmax
])

# Compile the model
model3.compile(
    loss=SparseCategoricalCrossentropy(),
    optimizer=Adam(),
    metrics=["accuracy"]
)
