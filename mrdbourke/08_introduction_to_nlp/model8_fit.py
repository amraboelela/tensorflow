from model8_init import *
   
tensorboard_path = "NLP/model8"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])


# Fit the model to 10% of the training data
history8 = model8.fit(
    x=train_sentences_10_percent,
    y=train_labels_10_percent,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(8)
    ]
)

# Save the history to a file
with open('data/history8.pkl', 'wb') as file:
    pickle.dump(history8.history, file)
