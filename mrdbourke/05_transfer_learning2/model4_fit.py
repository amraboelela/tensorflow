from model4_init import *

# Refit the model (same as model_2 except with more trainable layers)
history4 = model4.fit(
    train_data_10_percent,
    epochs=5,
    validation_data=test_data,
    validation_steps=int(0.25 * len(test_data)), # do less steps per validation (quicker)
    callbacks=[
        tensorboard_callback("transfer_learning/10_percent_data_aug"),
        checkpoint_callback(4)
    ]
)

model4.save_weights('data/model4.keras')

# Save the history to a file
with open('data/history4.pkl', 'wb') as file:
    pickle.dump(history4.history, file)

