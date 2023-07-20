from model2_init import *

# Fit model2 (this time we'll train for longer)
model2.fit(tf.expand_dims(X, axis=-1), y, epochs=100) # train for 100 epochs not 10

model2.save('data/model2.h5')
