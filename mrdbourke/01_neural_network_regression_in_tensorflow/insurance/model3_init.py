from common import *

# Set random seed
tf.random.set_seed(42)

# Build the model (3 layers, 100, 10, 1 units)
model3 = tf.keras.Sequential([
                              tf.keras.layers.Dense(100),
                              tf.keras.layers.Dense(10),
                              tf.keras.layers.Dense(1)
                              ])

# Compile the model
model3.compile(loss=tf.keras.losses.mae,
               optimizer=tf.keras.optimizers.Adam(),
               metrics=['mae'])
