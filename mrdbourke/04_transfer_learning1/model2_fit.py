from model2_init import *

# Fit EfficientNet model
history2 = model2.fit(train_data_10_percent, # only use 10% of training data
                      epochs=5, # train for 5 epochs
                      steps_per_epoch=len(train_data_10_percent),
                      validation_data=test_data,
                      validation_steps=len(test_data),
                      callbacks=[create_tensorboard_callback(dir_name="data/tensorflow_hub",
                                                             # Track logs under different experiment name
                                                             experiment_name="efficientnetB0")])

model2.save_weights('data/model2.keras')

# Save the history to a file
with open('data/history2.pkl', 'wb') as file:
    pickle.dump(history2.history, file)
