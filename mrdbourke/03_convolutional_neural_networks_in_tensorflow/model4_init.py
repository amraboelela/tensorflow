from common import *

# Create the model (this can be our baseline, a 3 layer Convolutional Neural Network)
model4 = Sequential([
  Conv2D(filters=10,
         kernel_size=3,
         strides=1,
         padding='valid',
         activation='relu',
         input_shape=(224, 224, 3)), # input layer (specify input shape)
  Conv2D(10, 3, activation='relu'),
  Conv2D(10, 3, activation='relu'),
  Flatten(),
  Dense(1, activation='sigmoid') # output layer (specify output shape)
])

# Compile the model
model4.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
