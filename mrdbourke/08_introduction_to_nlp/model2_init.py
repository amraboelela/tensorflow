from common import *

# Build model with the Functional API
x = GlobalAveragePooling1D()(x) # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
outputs = Dense(1, activation="sigmoid")(x) # create the output layer, want binary outputs so use sigmoid activation
model2 = tf.keras.Model(inputs, outputs, name="model2_dense") # construct the model

# Compile model
model2.compile(
    loss="binary_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"]
)
