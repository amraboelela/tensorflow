from common import *

# Create a model with a non-linear activation
model5 = Sequential([
    Dense(1, activation=relu), # can also do activation='relu'
    Dense(1) # output layer
])

# Compile the model
model5.compile(
    loss=binary_crossentropy,
    optimizer=Adam(),
    metrics=["accuracy"]
)
