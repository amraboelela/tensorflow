from common import *

# Set random seed
tf.random.set_seed(42)

# Create a model similar to model_1 but add an extra layer and increase the number of hidden units in each layer
model3 = Sequential([
    Flatten(input_shape=(224, 224, 3)), # dense layers expect a 1-dimensional vector as input
    Dense(100, activation='relu'), # increase number of neurons from 4 to 100 (for each layer)
    Dense(100, activation='relu'),
    Dense(100, activation='relu'), # add an extra layer
    Dense(1, activation='sigmoid')
])

# Compile the model
model3.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=["accuracy"]
)
