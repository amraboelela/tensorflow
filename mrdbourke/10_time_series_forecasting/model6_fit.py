from model6_init import *

# Fit
model6.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=128,
    verbose=0, # only print 1 line per epoch
    validation_data=(X_test, y_test),
    callbacks=[create_model_checkpoint(model_name=model6.name)]
)
