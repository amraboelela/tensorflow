from common import *

# Create model
model3 = create_model(model_url=efficientnet_url, # use EfficientNetB0 TensorFlow Hub URL
                      num_classes=train_data_10_percent.num_classes)

# Compile EfficientNet model
model3.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)
