from common import *

# Set random seed
tf.random.set_seed(42)

# Create a model (same as model_8)
model9 = Sequential([
    Dense(4, activation="relu"),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile the model
model9.compile(
    loss="binary_crossentropy", # we can use strings here too
    optimizer="Adam", # same as Adam() with default settings
    metrics=["accuracy"]
)
