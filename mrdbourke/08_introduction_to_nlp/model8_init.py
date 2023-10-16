from model7_init import *

# Clone model7 but reset weights
model8 = tf.keras.models.clone_model(model7)

# Compile model
model8.compile(
    loss="binary_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"]
)

