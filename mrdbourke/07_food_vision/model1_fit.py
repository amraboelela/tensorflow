from model1_init import *
   
tensorboard_path = "transfer_learning/model1"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Turn off all warnings except for errors
tf.get_logger().setLevel('ERROR')

# Fit the model with callbacks
history1 = model1.fit(
    train_data,
    epochs=3,
    steps_per_epoch=len(train_data),
    validation_data=test_data,
    validation_steps=int(0.15 * len(test_data)),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(1)
    ]
)

# Save the history to a file
with open('data/history1.pkl', 'wb') as file:
    pickle.dump(history1.history, file)

