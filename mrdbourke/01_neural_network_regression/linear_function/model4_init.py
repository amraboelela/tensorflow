from common2 import *

# Replicate model3 and add an extra layer
model4 = Sequential([
    Dense(100, activation="relu"),
    Dense(100, activation="relu"),
    Dense(1) # add a second layer
])

# Compile the model
model4.compile(
    loss=mae,
    optimizer=Adam(),
    metrics=['mae']
)

