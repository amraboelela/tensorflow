from model6_init import *

model6.load_weights(checkpoint_path(5))

history6 = model6.fit(
    train_data_all,
    epochs=5,
    validation_data=test_data_all,
    validation_steps=int(0.25 * len(test_data_10_percent)), # do less steps per validation (quicker)
    callbacks=[
        tensorboard_callback("transfer_learning/full_fine_tune_last_10"),
        checkpoint_callback(6)
    ]
)
                                           
# Save the history to a file
with open('data/history6.pkl', 'wb') as file:
    pickle.dump(history6.history, file)

