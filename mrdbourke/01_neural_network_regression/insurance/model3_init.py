from common import *

# Build the model (3 layers, 100, 10, 1 units)
model3 = Sequential([
    Dense(100),
    Dense(10),
    Dense(1)
])

# Compile the model
model3.compile(
    loss=mae,
    optimizer=Adam(),
    metrics=['mae']
)
