from model1_init import *
   
tensorboard_path = "transfer_learning/model1"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Fit
history1 = model1.fit(
    train_data_all_10_percent,
    epochs=5, # fit for 5 epochs to keep experiments quick
    validation_data=test_data,
    validation_steps=int(0.15 * len(test_data)), # evaluate on smaller portion of test data
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(1)
    ]
) # save best model weights to file

# Save the history to a file
with open('data/history1.pkl', 'wb') as file:
    pickle.dump(history1.history, file)

