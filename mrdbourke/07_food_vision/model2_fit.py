from model2_init import *
   
tensorboard_path = "transfer_learning/model2"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Start to fine-tune (all layers)
history2 = model2.fit(
    train_data,
    epochs=100, # fine-tune for a maximum of 100 epochs
    steps_per_epoch=len(train_data),
    validation_data=test_data,
    validation_steps=int(0.15 * len(test_data)), # validation during training on 15% of test data
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(2),
        early_stopping, # stop model after X epochs of no improvements
        reduce_lr
    ]
) # reduce the learning rate after X epochs of no improvements

# Save the history to a file
with open('data/history2.pkl', 'wb') as file:
    pickle.dump(history2.history, file)

