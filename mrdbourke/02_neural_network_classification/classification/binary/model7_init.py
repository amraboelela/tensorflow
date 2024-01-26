from common import *

# Create a model
model7 = Sequential([
    Dense(4, activation=relu), # hidden layer 1, ReLU activation
    Dense(4, activation=relu), # hidden layer 2, ReLU activation
    Dense(1, activation=sigmoid) # ouput layer, sigmoid activation
])

# Compile the model
model7.compile(
    loss=binary_crossentropy,
    optimizer=Adam(),
    metrics=['accuracy']
)
