from model1_init import *

# Fit the model
model1.fit(X_train, y_train, epochs=100)

model1.save('data/model1.h5')
