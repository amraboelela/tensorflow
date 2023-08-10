from model4_init import *

tensorboard_path = "transfer_learning/model4"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Refit the model (same as model_2 except with more trainable layers)
history4 = model4.fit(
    train_data_10_percent,
    epochs=5,
    validation_data=test_data,
    validation_steps=int(0.25 * len(test_data)), # do less steps per validation (quicker)
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(4)
    ]
)

model4.save_weights('data/model4.keras')

# Save the history to a file
with open('data/history4.pkl', 'wb') as file:
    pickle.dump(history4.history, file)

