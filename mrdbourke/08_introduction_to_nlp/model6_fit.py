from model6_init import *
   
tensorboard_path = "NLP/model6"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Fit model
history6 = model6.fit(
    train_sentences,
    train_labels,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(6)
    ]
)

# Save the history to a file
with open('data/history6.pkl', 'wb') as file:
    pickle.dump(history6.history, file)
