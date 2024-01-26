from model5_init import *

# Seems when saving the model several warnings are appearing: https://github.com/tensorflow/tensorflow/issues/47554
model5.fit(
    train_windows,
    train_labels,
    epochs=100,
    verbose=0,
    batch_size=128,
    validation_data=(test_windows, test_labels),
    callbacks=[create_model_checkpoint(model_name=model5.name)]
)
