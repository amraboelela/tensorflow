from common import *

# 1. Create the model (this time 3 layers)
model3 = Sequential([
    # Before TensorFlow 2.7.0
    # Dense(100), # add 100 dense neurons

    # With TensorFlow 2.7.0
    # Dense(100, input_shape=(None, 1)), # add 100 dense neurons

    ## After TensorFlow 2.8.0 ##
    Dense(100), # add 100 dense neurons
    Dense(10), # add another layer with 10 neurons
    Dense(1)
])

# 2. Compile the model
model3.compile(
    loss=BinaryCrossentropy(),
    optimizer=Adam(), # use Adam instead of SGD
    metrics=['accuracy']
)
