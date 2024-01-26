from common import *

# We can use this encoding layer in place of our text_vectorizer and embedding layer
sentence_encoder_layer = KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    input_shape=[], # shape of inputs coming to our model
    dtype=tf.string, # data type of inputs coming to the USE layer
    trainable=True,
    name="USE"
)

# Create model using the Sequential API
model9 = Sequential([
    sentence_encoder_layer, # take in sentences and then encode them into an embedding
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
], name="model9_USE")

# Compile model
model9.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

