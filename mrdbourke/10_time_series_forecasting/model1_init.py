from common import *

# Construct model
model1 = Sequential([
  layers.Dense(128, activation="relu"),
  layers.Dense(HORIZON, activation="linear") # linear activation is the same as having no activation
], name="model1_dense") # give the model a name so we can save it

# Compile model
model1.compile(
    loss="mae",
    optimizer=Adam(),
    metrics=["mae"]
) # we don't necessarily need this when the loss function is already MAE
