from model2_init import *

# Fit the model (to the normalized data)
history2 = model2.fit(train_data,
                      train_labels,
                      epochs=10,
                      validation_data=(test_data, test_labels))

model2.save('data/model2.keras')

# Save the history to a file
with open('data/history2.pkl', 'wb') as file:
    pickle.dump(history2.history, file)
