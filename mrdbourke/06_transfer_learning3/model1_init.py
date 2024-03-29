from common import *

# Setup base model and freeze its layers (this will extract features)
base_model = EfficientNetB0(include_top=False)
base_model.trainable = False

# Setup model architecture with trainable top layers
inputs = layers.Input(shape=(224, 224, 3), name="input_layer") # shape of input image
x = data_augmentation(inputs) # augment images (only happens during training)
x = base_model(x, training=False) # put the base model in inference mode so we can use it to extract features without updating the weights
x = GlobalAveragePooling2D(name="global_average_pooling")(x) # pool the outputs of the base model
outputs = Dense(len(train_data_all_10_percent.class_names), activation="softmax", name="output_layer")(x) # same number of outputs as classes
model1 = tf.keras.Model(inputs, outputs)

# Compile
model1.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(), # use Adam with default settings
    metrics=["accuracy"]
)
