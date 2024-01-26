from common import *

# Create the model
model4 = Sequential([
    Dense(1, activation=linear), # 1 hidden layer with linear activation
    Dense(1) # output layer
])

# Compile the model
model4.compile(
    loss=binary_crossentropy,
    optimizer=Adam(learning_rate=0.001), # note: "lr" used to be what was used, now "learning_rate" is favoured
    metrics=["accuracy"]
)
