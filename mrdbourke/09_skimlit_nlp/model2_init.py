from common import *

# Create 1D convolutional model to process sequences
inputs = layers.Input(shape=(1,), dtype=tf.string)
text_vectors = text_vectorizer(inputs) # vectorize text inputs
token_embeddings = token_embed(text_vectors) # create embedding
x = Conv1D(64, kernel_size=5, padding="same", activation="relu")(token_embeddings)
x = GlobalAveragePooling1D()(x) # condense the output of our feature vector
outputs = layers.Dense(num_classes, activation="softmax")(x)
model2 = tf.keras.Model(inputs, outputs)

# Compile
model2.compile(
    loss="categorical_crossentropy", # if your labels are integer form (not one hot) use sparse_categorical_crossentropy
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)
