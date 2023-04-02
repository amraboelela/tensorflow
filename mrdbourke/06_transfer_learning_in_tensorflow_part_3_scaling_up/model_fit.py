from model import *

# Fit
history_all_classes_10_percent = model.fit(train_data_all_10_percent,
                                           epochs=5, # fit for 5 epochs to keep experiments quick
                                           validation_data=test_data,
                                           validation_steps=int(0.15 * len(test_data)), # evaluate on smaller portion of test data
                                           callbacks=[checkpoint_callback]) # save best model weights to file

model.save_weights('data/model.h5')

# Save the history to a file
import pickle
with open('data/history_all_classes_10_percent.pkl', 'wb') as file:
    pickle.dump(history_all_classes_10_percent.history, file)

