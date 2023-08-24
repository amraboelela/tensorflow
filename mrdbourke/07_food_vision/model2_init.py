from common import *

# Load and evaluate downloaded GS model
model2 = tf.keras.models.load_model("data/07_efficientnetb0_feature_extract_model_mixed_precision")

# Are any of the layers in our model frozen?
for layer in model2.layers:
    layer.trainable = True # set all layers to trainable
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy) # make sure loaded model is using mixed precision dtype_policy ("mixed_float16")

# Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", # watch the val loss metric
    patience=3
) # if val loss decreases for 3 epochs in a row, stop training

# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
    patience=2,
    verbose=1, # print out when learning rate goes down
    min_lr=1e-7
)

# Compile the model
model2.compile(
    loss="sparse_categorical_crossentropy", # sparse_categorical_crossentropy for labels that are *not* one-hot
    optimizer=tf.keras.optimizers.Adam(0.0001), # 10x lower learning rate than the default
    metrics=["accuracy"]
)
