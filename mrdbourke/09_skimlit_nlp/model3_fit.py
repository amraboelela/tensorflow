from model3_init import *

tensorboard_path = "skimlit/model3"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Fit feature extractor model for 3 epochs
history3 = model3.fit(
    train_dataset,
    steps_per_epoch=int(0.1 * len(train_dataset)),
    epochs=3,
    validation_data=valid_dataset,
    validation_steps=int(0.1 * len(valid_dataset)),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(3)
    ]
)

# Save the history to a file
with open('data/history3.pkl', 'wb') as file:
    pickle.dump(history3.history, file)

