from common import *

# Setup input shape and base model, freezing the base model layers
input_shape = (224, 224, 3)
base_model = EfficientNetB0(include_top=False)
base_model.trainable = False

# Create input layer
inputs = layers.Input(shape=input_shape, name="input_layer")

# Add in data augmentation Sequential model as a layer
x = data_augmentation(inputs)

# Give base_model inputs (after augmentation) and don't train it
x = base_model(x, training=False)

# Pool output features of base model
x = GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

# Put a dense layer on as the output
outputs = Dense(10, activation="softmax", name="output_layer")(x)

# Make a model with inputs and outputs
model2 = tf.keras.Model(inputs, outputs)

# Compile the model
model2.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=["accuracy"]
)

