from model2_init import *

# Fit the model and save the history (we can plot this)
history2 = model2.fit(X_train, y_train, epochs=100, verbose=0)

model2.save('data/model2.h5')

# Save the history to a file
with open('data/history2.pkl', 'wb') as file:
    pickle.dump(history2.history, file)
