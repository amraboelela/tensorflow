from common import *

# Setup random seed
tf.random.set_seed(42)

# Recreate the model
model1 = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])

# Change the loss and metrics of our compiled model
model1.compile(loss=tf.keras.losses.mae, # change the loss function to be regression-specific
               optimizer=tf.keras.optimizers.Adam(),
               metrics=['mae']) # change the metric to be regression-specific
