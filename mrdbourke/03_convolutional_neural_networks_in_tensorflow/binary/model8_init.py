from common import *

# Create a CNN model (same as Tiny VGG but for binary classification - https://poloclub.github.io/cnn-explainer/ )
model8 = Sequential([
                     Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)), # same input shape as our images
                     Conv2D(10, 3, activation='relu'),
                     MaxPool2D(),
                     Conv2D(10, 3, activation='relu'),
                     Conv2D(10, 3, activation='relu'),
                     MaxPool2D(),
                     Flatten(),
                     Dense(1, activation='sigmoid')
                     ])

# Compile the model
model8.compile(loss="binary_crossentropy",
               optimizer=tf.keras.optimizers.Adam(),
               metrics=["accuracy"])
