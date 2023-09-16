from common2 import *

# Set random seed
tf.random.set_seed(42)

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

