from model4_init import *

model5 = tf.keras.models.clone_model(model4)

model5.layers[2].trainable = True

# Freeze all layers except for the top 10
for layer in model5.layers[2].layers[:-10]:
  layer.trainable = False
  
# Recompile the model (always recompile after any adjustments to a model)
model5.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # lr is 10x lower than before for fine-tuning
                metrics=["accuracy"])
