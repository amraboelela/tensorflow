from common import *

# Build model with the Functional API
inputs = layers.Input(shape=(1,), dtype="string") # inputs are 1-dimensional strings
x = text_vectorizer(inputs) # turn the input text into numbers
x = embedding(x) # create an embedding of the numerized numbers
x = GlobalAveragePooling1D()(x) # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
outputs = Dense(1, activation="sigmoid")(x) # create the output layer, want binary outputs so use sigmoid activation
model2 = tf.keras.Model(inputs, outputs, name="model2_dense") # construct the model

# Compile model
model2.compile(
    loss="binary_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"]
)
