from common2 import *

# Make multivariate time series model
model6 = Sequential([
        layers.Dense(128, activation="relu"),
        # layers.Dense(128, activation="relu"), # adding an extra layer here should lead to beating the naive model
        layers.Dense(HORIZON)
    ],
    name="model6_dense_multivariate"
)

# Compile
model6.compile(
    loss="mae",
    optimizer=Adam()
)
