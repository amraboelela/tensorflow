from model5_init import *

tensorboard_path = "skimlit/model5"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Fit the model on tokens and chars
history5 = model5.fit(
    train_char_token_dataset, # train on dataset of token and characters
    steps_per_epoch=int(0.1 * len(train_char_token_dataset)),
    epochs=3,
    validation_data=val_char_token_dataset,
    validation_steps=int(0.1 * len(val_char_token_dataset))
)

# Save the history to a file
with open('data/history5.pkl', 'wb') as file:
    pickle.dump(history5.history, file)
