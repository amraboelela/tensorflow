from common import *

# Add an extra layer and increase number of units
model2 = Sequential([
    Dense(100), # 100 units
    Dense(10), # 10 units
    Dense(1) # 1 unit (important for output layer)
])

# Compile the model
model2.compile(
    loss=mae,
    optimizer=Adam(), # Adam works but SGD doesn't
    metrics=['mae']
)
