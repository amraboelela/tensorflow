from model3_init import *

# Fit the model
history3 = model3.fit(train_data_augmented, # use augmented data
                      epochs=5,
                      steps_per_epoch=len(train_data_augmented),
                      validation_data=test_data,
                      validation_steps=len(test_data))

model3.save_weights('data/model3.keras')

# Save the history to a file
with open('data/history3.pkl', 'wb') as file:
    pickle.dump(history3.history, file)

