from model1_init import *

# Fit the model
# model.fit(X, y, epochs=5) # this will break with TensorFlow 2.7.0+
model1.fit(tf.expand_dims(X, axis=-1), y, epochs=5)

model1.save('data/model1.keras')
