from common import *

# Set random seed
tf.random.set_seed(42)

# 1. Create the model using the Sequential API
model1 = Sequential([
    Dense(1)
])

# 2. Compile the model
model1.compile(
    loss=BinaryCrossentropy(), # binary since we are working with 2 clases (0 & 1)
    optimizer=SGD(),
    metrics=['accuracy']
)
