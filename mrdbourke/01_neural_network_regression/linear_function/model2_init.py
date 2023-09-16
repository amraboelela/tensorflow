from common1 import *

# Set random seed
tf.random.set_seed(42)

# Create a model2 (same as above)
model2 = Sequential([
    Dense(1)
])

# Compile model2 (same as above)
model2.compile(
    loss=mae,
    optimizer=SGD(),
    metrics=["mae"]
)
