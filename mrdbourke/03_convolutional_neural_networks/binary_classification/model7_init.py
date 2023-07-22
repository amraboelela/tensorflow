from common import *

# Create the model (same as model_5)
model7 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  MaxPool2D(pool_size=2), # reduce number of features by half
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(1, activation='sigmoid')
])

# Compile the model
model7.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

# model7 is built over model6
model7.load_weights('data/model6.keras')
