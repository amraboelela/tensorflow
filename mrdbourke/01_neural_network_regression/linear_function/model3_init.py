from common2 import *

# Set random seed
tf.random.set_seed(42)

# Create a model (same as above)
#model3 = tf.keras.Sequential([
#  tf.keras.layers.Dense(1, input_shape=[1]) # define the input_shape to our model
#])

# Replicate original model
model3 = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile model (same as above)
model3.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])
