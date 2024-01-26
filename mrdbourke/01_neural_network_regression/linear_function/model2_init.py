from common1 import *

# Create a model2 (same as above)
model2 = Sequential([
    Dense(1)
])

# Compile model2 (same as above)
model2.compile(
    loss=mae,
    optimizer=SGD(),
    metrics=["mae"]
)
