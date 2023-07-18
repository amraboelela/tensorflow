from common1 import *

# Set random seed
tf.random.set_seed(42)

# Create a model using the Sequential API
model1 = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile the model
model1.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
              optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
              metrics=["mae"])
