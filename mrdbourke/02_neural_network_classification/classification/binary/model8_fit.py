from model8_init import *

# Fit the model
history8 = model8.fit(X_train, y_train, epochs=75)

model8.save('data/model8.keras')

# Save the history to a file
with open('data/history8.pkl', 'wb') as file:
    pickle.dump(history8.history, file)
