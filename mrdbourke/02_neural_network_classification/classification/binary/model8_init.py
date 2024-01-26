from common import *

# Create the model (same as model_7)
model8 = Sequential([
    Dense(4, activation="relu"), # hidden layer 1, using "relu" for activation (same as relu)
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid") # output layer, using 'sigmoid' for the output
])

# Compile the model
model8.compile(
    loss=binary_crossentropy,
    optimizer=Adam(lr=0.01), # increase learning rate from 0.001 to 0.01 for faster learning
    metrics=['accuracy']
)
