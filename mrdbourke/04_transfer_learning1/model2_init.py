from common import *

# Create model
model2 = create_model(model_url=efficientnet_url, # use EfficientNetB0 TensorFlow Hub URL
                      num_classes=train_data_10_percent.num_classes)

# Compile EfficientNet model
model2.compile(loss='categorical_crossentropy',
               optimizer=tf.keras.optimizers.Adam(),
               metrics=['accuracy'])
