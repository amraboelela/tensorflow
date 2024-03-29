from model8_init import *
                        
# Fit the model
history8 = model8.fit(train_data_augmented_shuffled,
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented_shuffled),
                        validation_data=test_data,
                        validation_steps=len(test_data))

model8.save_weights('data/model8.keras')

# Save the history to a file
with open('data/history8.pkl', 'wb') as file:
    pickle.dump(history8.history, file)
