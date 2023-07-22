from model6_init import *

# Fit the model
history6 = model6.fit(X, y, epochs=100)

model6.save('data/model6.keras')

# Save the history to a file
with open('data/history6.pkl', 'wb') as file:
    pickle.dump(history6.history, file)
