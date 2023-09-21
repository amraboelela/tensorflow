from model5_init import *
   
tensorboard_path = "NLP/model5"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Fit model
history5 = model5.fit(
    train_sentences,
    train_labels,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(5)
    ]
)

# Save the history to a file
with open('data/history5.pkl', 'wb') as file:
    pickle.dump(history5.history, file)
