from model2 import *

# Fine-tune for 5 more epochs
fine_tune_epochs = 10 # model has already done 5 epochs, this is the total number of epochs we're after (5+5=10)

history_all_classes_10_percent_fine_tune = model.fit(train_data_all_10_percent,
                                                     epochs=fine_tune_epochs,
                                                     validation_data=test_data,
                                                     validation_steps=int(0.15 * len(test_data)), # validate on 15% of the test data
                                                     initial_epoch=history_all_classes_10_percent.epoch[-1]) # start from previous last epoch

model.save_weights('data/model2.h5')

# Save the history to a file
import pickle
with open('data/history_all_classes_10_percent_fine_tune.pkl', 'wb') as file:
    pickle.dump(history_all_classes_10_percent_fine_tune.history, file)

