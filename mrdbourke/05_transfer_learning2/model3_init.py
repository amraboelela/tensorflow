from model2_init import *

model3 = tf.keras.models.clone_model(model2)

# Compile the model
model3.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
