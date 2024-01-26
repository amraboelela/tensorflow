from common import *

# Create the model
model10 = Sequential([
    Dense(4, activation="relu"),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile the model with the ideal learning rate
model10.compile(
    loss="binary_crossentropy",
    optimizer=Adam(lr=0.02), # to adjust the learning rate, you need to use Adam (not "adam")
    metrics=["accuracy"]
)
