from model3_init import *

model3.load_weights('data/model2.keras')

# Fit the model
history3 = model3.fit(
    train_data_1_percent,
    epochs=5,
    steps_per_epoch=len(train_data_1_percent),
    validation_data=test_data,
    validation_steps=int(0.25* len(test_data)), # validate for less steps
    # Track model training logs
    callbacks=[tensorboard_callback("transfer_learning/1_percent_data_aug")]
)

model3.save_weights('data/model3.keras')

# Save the history to a file
with open('data/history3.pkl', 'wb') as file:
    pickle.dump(history3.history, file)

