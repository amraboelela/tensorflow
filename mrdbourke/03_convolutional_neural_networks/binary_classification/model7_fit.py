from model7_init import *
               
# Fit the model
history7 = model7.fit(train_data_augmented_shuffled, # now the augmented data is shuffled
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented_shuffled),
                        validation_data=test_data,
                        validation_steps=len(test_data))

model7.save_weights('data/model7.keras')

# Save the history to a file
with open('data/history7.pkl', 'wb') as file:
    pickle.dump(history7.history, file)
