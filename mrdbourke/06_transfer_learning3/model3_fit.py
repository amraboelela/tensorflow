from model2_init import *

# Fine-tune for 5 more epochs
fine_tune_epochs = 10 # model has already done 5 epochs, this is the total number of epochs we're after (5+5=10)

tensorboard_path = "transfer_learning/model3"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

history3 = model3.fit(
    train_data_all_10_percent,
    epochs=fine_tune_epochs,
    validation_data=test_data,
    validation_steps=int(0.15 * len(test_data)), # validate on 15% of the test data
    initial_epoch=5,
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(3)
    ]
) # start from previous last epoch

# Save the history to a file
with open('data/history3.pkl', 'wb') as file:
    pickle.dump(history3.history, file)

