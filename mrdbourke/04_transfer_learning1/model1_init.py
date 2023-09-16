from common import *

# Create model
model1 = create_model(resnet_url, num_classes=train_data_10_percent.num_classes)

# Compile
model1.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)
