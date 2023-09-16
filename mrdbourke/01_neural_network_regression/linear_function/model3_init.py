from common2 import *

# Set random seed
tf.random.set_seed(42)

# Replicate original model
model3 = Sequential([
    Dense(1)
])

# Compile model (same as above)
model3.compile(
    loss=mae,
    optimizer=SGD(),
    metrics=["mae"]
)
