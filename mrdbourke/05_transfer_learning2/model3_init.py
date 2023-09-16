from model2_init import *

model3 = clone_model(model2)

# Compile the model
model3.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=["accuracy"]
)
