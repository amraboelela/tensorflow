from common1 import *

# Set random seed
tf.random.set_seed(42)

# Create a model using the Sequential API
model1 = Sequential([
    Dense(1)
])

# Compile the model
model1.compile(
    loss=mae, # mae is short for mean absolute error
    optimizer=SGD(), # SGD is short for stochastic gradient descent
    metrics=["mae"]
)
