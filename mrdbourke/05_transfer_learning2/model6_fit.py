from model6_init import *

history6 = model6.fit(
    train_data_all,
    epochs=5,
    validation_data=test_data,
    validation_steps=int(0.25 * len(test_data)), # do less steps per validation (quicker)
    callbacks=[
        tensorboard_callback("transfer_learning/full_fine_tune_last_10"),
        checkpoint_callback(6)
    ]
)
                                           
# Save the history to a file
with open('data/history6.pkl', 'wb') as file:
    pickle.dump(history6.history, file)

