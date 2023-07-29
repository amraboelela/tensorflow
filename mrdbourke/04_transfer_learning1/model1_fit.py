from model1_init import *

# Fit the model
history1 = model1.fit(train_data_10_percent,
                      epochs=5,
                      steps_per_epoch=len(train_data_10_percent),
                      validation_data=test_data,
                      validation_steps=len(test_data),
                      # Add TensorBoard callback to model (callbacks parameter takes a list)
                      callbacks=[create_tensorboard_callback(dir_name="data/tensorflow_hub", # save experiment logs here
                                                             experiment_name="resnet50V2")]) # name of log files

model1.save_weights('data/model1.keras')

# Save the history to a file
with open('data/history1.pkl', 'wb') as file:
    pickle.dump(history1.history, file)

