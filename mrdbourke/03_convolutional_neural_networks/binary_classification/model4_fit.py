from model4_init import *

# Fit the model
history4 = model4.fit(train_data,
                      epochs=5,
                      steps_per_epoch=len(train_data),
                      validation_data=test_data,
                      validation_steps=len(test_data))

model4.save_weights('data/model4.keras')

# Save the history to a file
with open('data/history4.pkl', 'wb') as file:
    pickle.dump(history4.history, file)
