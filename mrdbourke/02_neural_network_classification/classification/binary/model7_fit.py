from model7_init import *

# Fit the model
history7 = model7.fit(X, y, epochs=100, verbose=0)

model7.save('data/model7.h5')

# Save the history to a file
with open('data/history7.pkl', 'wb') as file:
    pickle.dump(history7.history, file)
