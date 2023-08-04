from model5_init import *

model6 = tf.keras.models.clone_model(model5)

model6.load_weights(checkpoint_path(5))

# Recompile the model (always recompile after any adjustments to a model)
model6.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # lr is 10x lower than before for fine-tuning
    metrics=["accuracy"]
)
