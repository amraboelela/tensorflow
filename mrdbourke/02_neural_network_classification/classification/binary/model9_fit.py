from model9_init import *

# Create a learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20)) # traverse a set of learning rate values starting from 1e-4, increasing by 10**(epoch/20) every epoch

# Fit the model (passing the lr_scheduler callback)
history9 = model9.fit(X_train,
                      y_train,
                      epochs=100,
                      callbacks=[lr_scheduler])

model9.save('data/model9.keras')

# Save the history to a file
with open('data/history9.pkl', 'wb') as file:
    pickle.dump(history9.history, file)
