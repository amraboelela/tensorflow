from common import *

# Set random seed
tf.random.set_seed(42)

# Build the model (3 layers, 100, 10, 1 units)
model3 = Sequential([
    Dense(100),
    Dense(10),
    Dense(1)
])

# Compile the model
model3.compile(
    loss=mae,
    optimizer=Adam(),
    metrics=['mae']
)
