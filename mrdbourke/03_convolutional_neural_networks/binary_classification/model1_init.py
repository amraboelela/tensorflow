from common import *

# Create a CNN model (same as Tiny VGG - https://poloclub.github.io/cnn-explainer/)
model1 = Sequential([
    Conv2D(
        filters=10,
        kernel_size=3, # can also be (3, 3)
        activation="relu",
        input_shape=(224, 224, 3)
    ), # first layer specifies input shape (height, width, colour channels)
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(
        pool_size=2, # pool_size can also be (2, 2)
        padding="valid"
    ), # padding can also be 'same'
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"), # activation='relu' == Activations(tf.nn.relu)
    MaxPool2D(2),
    Flatten(),
    Dense(1, activation="sigmoid") # binary activation output
])

# Compile the model
model1.compile(
    loss="binary_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"]
)
