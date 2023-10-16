from common import *

# Build a Bidirectional RNN in TensorFlow
# x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x) # stacking RNN layers requires return_sequences=True
x = Bidirectional(LSTM(64))(x) # bidirectional goes both ways so has double the parameters of a regular LSTM layer
outputs = Dense(1, activation="sigmoid")(x)
model5 = tf.keras.Model(inputs, outputs, name="model5_Bidirectional")

# Compile
model5.compile(
    loss="binary_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"]
)
