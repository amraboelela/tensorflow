from model4_init import *

# Fit the model
history4 = model4.fit(train_data,
                      train_labels,
                      epochs=20,
                      validation_data=(test_data, test_labels))

model4.save('data/model4.keras')

# Save the history to a file
with open('data/history4.pkl', 'wb') as file:
    pickle.dump(history4.history, file)
