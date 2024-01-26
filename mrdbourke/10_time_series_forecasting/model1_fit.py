from model1_init import *

# Fit model
model1.fit(
    x=train_windows, # train windows of 7 timesteps of Bitcoin prices
    y=train_labels, # horizon value of 1 (using the previous 7 timesteps to predict next day)
    epochs=100,
    verbose=1,
    batch_size=128,
    validation_data=(test_windows, test_labels),
    callbacks=[create_model_checkpoint(model_name=model1.name)]
) # create ModelCheckpoint callback to save best model
