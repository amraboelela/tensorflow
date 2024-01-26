from model4_init import *

# Fit model
model4.fit(
    train_windows,
    train_labels,
    batch_size=128,
    epochs=100,
    verbose=0,
    validation_data=(test_windows, test_labels),
    callbacks=[create_model_checkpoint(model_name=model4.name)]
)
