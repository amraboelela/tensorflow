from model3_init import *

# Fit the model to the training data
#model3.fit(X_train, y_train, epochs=100, verbose=0) # verbose controls how much gets output

model3.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)

model3.save('data/model3.keras')
