from common import *

# Set random seed
tf.random.set_seed(42)

# Create a model to replicate the TensorFlow Playground model
model2 = Sequential([
    Flatten(input_shape=(224, 224, 3)), # dense layers expect a 1-dimensional vector as input
    Dense(4, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model2.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=["accuracy"]
)
