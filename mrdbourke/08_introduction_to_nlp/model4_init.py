from common import *

# Build an RNN using the GRU cell
# x = layers.GRU(64, return_sequences=True) # stacking recurrent cells requires return_sequences=True
x = GRU(64)(x)
# x = Dense(64, activation="relu")(x) # optional dense layer after GRU cell
outputs = Dense(1, activation="sigmoid")(x)
model4 = tf.keras.Model(inputs, outputs, name="model4_GRU")

# Compile GRU model
model4.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

