from common import *

# Create LSTM model
# x = layers.LSTM(64, return_sequences=True)(x) # return vector for each word in the Tweet (you can stack RNN cells as long as return_sequences=True)
x = LSTM(64)(x) # return vector for whole sequence
print(x.shape)
# x = Dense(64, activation="relu")(x) # optional dense layer on top of output of LSTM cell
outputs = Dense(1, activation="sigmoid")(x)
model3 = tf.keras.Model(inputs, outputs, name="model3_LSTM")

# Compile model
model3.compile(
    loss="binary_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"]
)
