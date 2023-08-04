from model1_init import *

# 10. Fit the model (we use less steps for validation so it's faster)
history1 = model1.fit(
    train_data_10_percent,
    epochs=5,
    steps_per_epoch=len(train_data_10_percent),
    validation_data=test_data, # Go through less of the validation data so epochs are faster (we want faster experiments!)
    validation_steps=int(0.25 * len(test_data)), # Track our model's training logs for visualization later
    callbacks=[tensorboard_callback("transfer_learning/10_percent_feature_extract")]
)

model1.save_weights('data/model1.keras')

# Save the history to a file
with open('data/history1.pkl', 'wb') as file:
    pickle.dump(history1.history, file)

