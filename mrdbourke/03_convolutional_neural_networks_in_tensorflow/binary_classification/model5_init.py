from common import *

# Create the model (this can be our baseline, a 3 layer Convolutional Neural Network)
model5 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  MaxPool2D(pool_size=2), # reduce number of features by half
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(1, activation='sigmoid')
])

# Compile model (same as model_4)
model5.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
