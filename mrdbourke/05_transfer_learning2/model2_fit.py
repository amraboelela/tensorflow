from model2_init import *

# Fit the model
history2 = model2.fit(train_data_1_percent,
                      epochs=5,
                      steps_per_epoch=len(train_data_1_percent),
                      validation_data=test_data,
                      validation_steps=int(0.25* len(test_data)), # validate for less steps
                      # Track model training logs
                      callbacks=[tensorboard_callback("transfer_learning/1_percent_data_aug")])

model2.save_weights('data/model2.keras')

# Save the history to a file
with open('data/history2.pkl', 'wb') as file:
    pickle.dump(history2.history, file)

