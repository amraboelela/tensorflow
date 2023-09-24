from common import *

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

