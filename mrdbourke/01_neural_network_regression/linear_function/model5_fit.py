from model5_init import *

# Fit the model (this time for 500 epochs, not 100)
model5.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=500, verbose=0) # set verbose to 0 for less output

model5.save('data/model5.keras')
