from model7_init import *

# 9. Fit the model with EarlyStopping and ReduceLROnPlateau callbacks
model7.fit(
    train_dataset,
    epochs=N_EPOCHS,
    validation_data=test_dataset,
    verbose=0, # prevent large amounts of training outputs
            # callbacks=[create_model_checkpoint(model_name=stack_model.name)] # saving model every epoch consumes far too much time
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)
    ]
)
