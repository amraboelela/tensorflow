from model3_init import *

# Create the learning rate callback
lr_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

# Fit the model
history3 = model3.fit(
    train_data,
    train_labels,
    epochs=40, # model already doing pretty good with current LR, probably don't need 100 epochs
    validation_data=(test_data, test_labels),
    callbacks=[lr_scheduler]
)

model3.save('data/model3.keras')

# Save the history to a file
with open('data/history3.pkl', 'wb') as file:
    pickle.dump(history3.history, file)
