from model4_init import *
   
tensorboard_path = "NLP/model4"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Fit model
history4 = model4.fit(
    train_sentences,
    train_labels,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(4)
    ]
)

# Save the history to a file
with open('data/history4.pkl', 'wb') as file:
    pickle.dump(history4.history, file)
