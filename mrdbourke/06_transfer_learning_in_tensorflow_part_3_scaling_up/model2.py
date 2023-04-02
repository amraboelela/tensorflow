from model_load import *

# Unfreeze all of the layers in the base model
base_model.trainable = True

# Refreeze every layer except for the last 5
for layer in base_model.layers[:-5]:
  layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4), # 10x lower learning rate than default
              metrics=['accuracy'])

# What layers in the model are trainable?
#for layer in model.layers:
#  print(layer.name, layer.trainable)

# Check which layers are trainable
#for layer_number, layer in enumerate(base_model.layers):
#  print(layer_number, layer.name, layer.trainable)

