from common2 import *

# Replicate original model
model3 = Sequential([
    Dense(1)
])

# Compile model (same as above)
model3.compile(
    loss=mae,
    optimizer=SGD(),
    metrics=["mae"]
)
