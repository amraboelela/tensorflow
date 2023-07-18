from common1 import *

# Set random seed
tf.random.set_seed(42)

# Create a model2 (same as above)
model2 = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile model2 (same as above)
model2.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])
