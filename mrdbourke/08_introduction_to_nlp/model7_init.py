from common import *

# We can use this encoding layer in place of our text_vectorizer and embedding layer
sentence_encoder_layer = KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    input_shape=[], # shape of inputs coming to our model
    dtype=tf.string, # data type of inputs coming to the USE layer
    trainable=False, # keep the pretrained weights (we'll create a feature extractor)
    name="USE"
)

# Create model using the Sequential API
model7 = Sequential([
    sentence_encoder_layer, # take in sentences and then encode them into an embedding
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
], name="model7_USE")

# Compile model
model7.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

