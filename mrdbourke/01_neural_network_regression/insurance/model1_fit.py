from model1_init import *

# Fit the model
model1.fit(X_train_oh, y_train_oh, epochs=100)

model1.save('data/model1.keras')
