from common import *

# Make Conv1D on chars only
inputs = layers.Input(shape=(1,), dtype="string")
char_vectors = char_vectorizer(inputs)
char_embeddings = char_embed(char_vectors)
x = Conv1D(64, kernel_size=5, padding="same", activation="relu")(char_embeddings)
x = GlobalMaxPool1D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model4 = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="model4_conv1D_char_embedding"
)

# Compile model
model4.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"]
)

