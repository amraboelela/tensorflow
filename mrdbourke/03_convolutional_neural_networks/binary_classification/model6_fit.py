from model6_init import *
                        
# Fit the model
history6 = model6.fit(train_data_augmented, # changed to augmented training data
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=test_data,
                        validation_steps=len(test_data))

model6.save_weights('data/model6.h5')

# Save the history to a file
with open('data/history6.pkl', 'wb') as file:
    pickle.dump(history6.history, file)
