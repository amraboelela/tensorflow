from common import *

# Setup input shape and base model, freezing the base model layers
input_shape = (224, 224, 3)

# Create a frozen base model
base_model = EfficientNetB0(include_top=False)
base_model.trainable = False

# Create input and output layers
inputs = layers.Input(shape=input_shape, name="input_layer") # create input layer
x = data_augmentation(inputs) # augment our training images
x = base_model(x, training=False) # pass augmented images to base model but keep it in inference mode, so batchnorm layers don't get updated: https://keras.io/guides/transfer_learning/#build-a-model
x = GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
outputs = Dense(10, activation="softmax", name="output_layer")(x)
model4 = tf.keras.Model(inputs, outputs)

# Compile
model4.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001), # use Adam optimizer with base learning rate
    metrics=["accuracy"]
)
