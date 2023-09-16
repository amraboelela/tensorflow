from model2_init import *

# Clone the model (use the same architecture)
model3 = clone_model(model2)

# Compile the cloned model (same setup as used for model_10)
model3.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"]
)
