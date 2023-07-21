from model1_init import *

# Fit the model
history1 = model1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))

model1.save_weights('data/model1.h5')

# Save the history to a file
with open('data/history1.pkl', 'wb') as file:
    pickle.dump(history1.history, file)

