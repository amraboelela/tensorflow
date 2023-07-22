from common import *

# Set random seed
tf.random.set_seed(42)

# Create the model (same as model_7)
model8 = tf.keras.Sequential([
                              tf.keras.layers.Dense(4, activation="relu"), # hidden layer 1, using "relu" for activation (same as tf.keras.activations.relu)
                              tf.keras.layers.Dense(4, activation="relu"),
                              tf.keras.layers.Dense(1, activation="sigmoid") # output layer, using 'sigmoid' for the output
                              ])

# Compile the model
model8.compile(loss=tf.keras.losses.binary_crossentropy,
               optimizer=tf.keras.optimizers.Adam(lr=0.01), # increase learning rate from 0.001 to 0.01 for faster learning
               metrics=['accuracy'])
