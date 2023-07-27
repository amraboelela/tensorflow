from model2_init import *

# Clone the model (use the same architecture)
model3 = tf.keras.models.clone_model(model2)

# Compile the cloned model (same setup as used for model_10)
model3.compile(loss="categorical_crossentropy",
               optimizer=tf.keras.optimizers.Adam(),
               metrics=["accuracy"])
