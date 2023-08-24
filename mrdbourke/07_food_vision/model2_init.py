from common import *

# Load and evaluate downloaded GS model
model2 = tf.keras.models.load_model("data/07_efficientnetb0_feature_extract_model_mixed_precision")

# Are any of the layers in our model frozen?
for layer in model2.layers:
    layer.trainable = True # set all layers to trainable
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy) # make sure loaded model is using mixed precision dtype_policy ("mixed_float16")
