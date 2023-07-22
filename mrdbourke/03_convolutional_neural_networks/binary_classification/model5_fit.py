from model5_init import *

# Fit the model
history5 = model5.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))

model5.save_weights('data/model5.keras')

# Save the history to a file
with open('data/history5.pkl', 'wb') as file:
    pickle.dump(history5.history, file)
