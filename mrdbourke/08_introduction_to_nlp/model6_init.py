from common import *

x = Conv1D(filters=32, kernel_size=5, activation="relu")(x)
x = GlobalMaxPool1D()(x)
# x = Dense(64, activation="relu")(x) # optional dense layer
outputs = Dense(1, activation="sigmoid")(x)
model6 = tf.keras.Model(inputs, outputs, name="model6_Conv1D")

# Compile Conv1D model
model6.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)
