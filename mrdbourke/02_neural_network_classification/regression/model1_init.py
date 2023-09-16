from common import *

# Setup random seed
tf.random.set_seed(42)

# Recreate the model
model1 = Sequential([
    Dense(100),
    Dense(10),
    Dense(1)
])

# Change the loss and metrics of our compiled model
model1.compile(
    loss=mae, # change the loss function to be regression-specific
    optimizer=Adam(),
    metrics=['mae']
) # change the metric to be regression-specific
