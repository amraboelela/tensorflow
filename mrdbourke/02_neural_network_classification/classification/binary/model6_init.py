from common import *

# Create a model
model6 = Sequential([
    Dense(4, activation=relu), # hidden layer 1, 4 neurons, ReLU activation
    Dense(4, activation=relu), # hidden layer 2, 4 neurons, ReLU activation
    Dense(1) # ouput layer
])

# Compile the model
model6.compile(
    loss=binary_crossentropy,
    optimizer=Adam(lr=0.001), # Adam's default learning rate is 0.001
    metrics=['accuracy']
)
