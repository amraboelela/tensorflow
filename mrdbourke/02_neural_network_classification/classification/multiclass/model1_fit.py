from model1_init import *

# Fit the model
history1 = model1.fit(train_data,
                      train_labels,
                      epochs=10,
                      validation_data=(test_data, test_labels)) # see how the model performs on the test set during training

model1.save('data/model1.keras')

# Save the history to a file
with open('data/history1.pkl', 'wb') as file:
    pickle.dump(history1.history, file)
