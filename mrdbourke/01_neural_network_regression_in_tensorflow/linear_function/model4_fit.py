from model4_init import *

# Fit the model
model4.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=500, verbose=1) # set verbose to 0 for less output

model4.save('data/model4.h5')
