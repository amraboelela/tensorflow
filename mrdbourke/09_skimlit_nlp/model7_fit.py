from model6_init import *

tensorboard_path = "skimlit/model6"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Fit the token, char and positional embedding model
history6 = model6.fit(
    train_pos_char_token_dataset,
    steps_per_epoch=int(0.1 * len(train_pos_char_token_dataset)),
    epochs=3,
    validation_data=val_pos_char_token_dataset,
    validation_steps=int(0.1 * len(val_pos_char_token_dataset)),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(6)
    ]
)

# Save the history to a file
with open('data/history6.pkl', 'wb') as file:
    pickle.dump(history6.history, file)
