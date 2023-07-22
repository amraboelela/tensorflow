from model4_init import *

# Fit the model
history4 = model4.fit(X, y, epochs=100)

model4.save('data/model4.h5')

# Save the history to a file
with open('data/history4.pkl', 'wb') as file:
    pickle.dump(history4.history, file)
