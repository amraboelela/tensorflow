from common import *

# Set random seed
tf.random.set_seed(42)

# Create a new model (same as model_2)
model1 = Sequential([
    Dense(1),
    Dense(1)
])

# Compile the model
model1.compile(
    loss=mae,
    optimizer=SGD(),
    metrics=['mae']
)
