from model1_init import *

tensorboard_path = "transfer_learning/model1"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Fit the model
history1 = model1.fit(train_data_10_percent,
                      epochs=5,
                      steps_per_epoch=len(train_data_10_percent),
                      validation_data=test_data,
                      validation_steps=len(test_data),
                      # Add TensorBoard callback to model (callbacks parameter takes a list)
                      callbacks=[tensorboard_callback(tensorboard_path)]) # name of log files

model1.save_weights('data/model1.keras')

# Save the history to a file
with open('data/history1.pkl', 'wb') as file:
    pickle.dump(history1.history, file)

