from model2_init import *

# Fit the model
history2 = model2.fit(train_data, # use same training data created above
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data, # use same validation data created above
                        validation_steps=len(valid_data))
                        
model2.save_weights('data/model2.h5')

# Save the history to a file
with open('data/history2.pkl', 'wb') as file:
    pickle.dump(history2.history, file)
