from model5_init import *

# Fit the model
history5 = model5.fit(X, y, epochs=100)

model5.save('data/model5.h5')

# Save the history to a file
with open('data/history5.pkl', 'wb') as file:
    pickle.dump(history5.history, file)
