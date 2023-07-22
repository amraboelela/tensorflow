from model10_init import *

# Fit the model for 20 epochs (5 less than before)
history10 = model10.fit(X_train, y_train, epochs=75)

model10.save('data/model10.keras')

# Save the history to a file
with open('data/history10.pkl', 'wb') as file:
    pickle.dump(history10.history, file)
