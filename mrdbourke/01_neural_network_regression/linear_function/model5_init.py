from common2 import *

# Set random seed
tf.random.set_seed(42)

# Replicate model4
model5 = Sequential([
    Dense(1),
    Dense(1)
])

# Compile the model
model5.compile(
    loss=mae,
    optimizer=SGD(),
    metrics=['mae']
)
