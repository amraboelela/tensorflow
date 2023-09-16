from common import *

# Set random seed
tf.random.set_seed(42)

# 1. Create the model (same as model_1 but with an extra layer)
model2 = Sequential([
    Dense(1), # add an extra layer
    Dense(1)
])

# 2. Compile the model
model2.compile(
    loss=BinaryCrossentropy(),
    optimizer=SGD(),
    metrics=['accuracy']
)
