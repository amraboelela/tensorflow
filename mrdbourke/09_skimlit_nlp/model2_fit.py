from model2_init import *

tensorboard_path = "skimlit/model2"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Fit the model
history2 = model2.fit(
    train_dataset,
    steps_per_epoch=int(0.1 * len(train_dataset)), # only fit on 10% of batches for faster training time
    epochs=3,
    validation_data=valid_dataset,
    validation_steps=int(0.1 * len(valid_dataset)),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(2)
    ]
) # only validate on 10% of batches

# Save the history to a file
with open('data/history2.pkl', 'wb') as file:
    pickle.dump(history2.history, file)

