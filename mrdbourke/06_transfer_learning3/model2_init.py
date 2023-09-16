from model1_init import *

model2 = clone_model(model1)

model2.load_weights(checkpoint_path(1))

model2.layers[2].trainable = True

# Refreeze every layer except for the last 5
for layer in model2.layers[2].layers[:-5]:
    layer.trainable = False
  
# Recompile model with lower learning rate
model2.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-4), # 10x lower learning rate than default
    metrics=['accuracy']
)
