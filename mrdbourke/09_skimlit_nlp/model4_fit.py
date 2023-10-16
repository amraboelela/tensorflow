from model4_init import *

tensorboard_path = "skimlit/model4"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Fit the model on chars only
history4 = model4.fit(
    train_char_dataset,
    steps_per_epoch=int(0.1 * len(train_char_dataset)),
    epochs=3,
    validation_data=val_char_dataset,
    validation_steps=int(0.1 * len(val_char_dataset)),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(4)
    ]
)

# Save the history to a file
with open('data/history4.pkl', 'wb') as file:
    pickle.dump(history4.history, file)
