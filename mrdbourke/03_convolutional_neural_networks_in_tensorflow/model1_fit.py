from model1_init import *

# Fit the model
history_1 = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

model.save_weights('data/model1.h5')

# Save the history to a file
with open('data/history_1.pkl', 'wb') as file:
    pickle.dump(history_1.history, file)

