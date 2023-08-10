from model5_init import *

model5.load_weights(checkpoint_path(4))

tensorboard_path = "transfer_learning/model5"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Refit the model (same as model4 except with more trainable layers)
history5 = model5.fit(
    train_data_10_percent,
    epochs=5,
    validation_data=test_data,
    validation_steps=int(0.25 * len(test_data)), # do less steps per validation (quicker)
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(5)
    ]
)

# Save the history to a file
with open('data/history5.pkl', 'wb') as file:
    pickle.dump(history5.history, file)

