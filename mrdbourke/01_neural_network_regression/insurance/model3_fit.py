from model3_init import *

# Fit the model for 200 epochs (same as insurance_model_2)
history3 = model3.fit(X_train_normal, y_train, epochs=200, verbose=0)

model3.save('data/model3.keras')

# Save the history to a file
with open('data/history3.pkl', 'wb') as file:
    pickle.dump(history3.history, file)
