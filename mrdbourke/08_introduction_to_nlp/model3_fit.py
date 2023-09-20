from model3_init import *
   
tensorboard_path = "NLP/model3"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Fit model
history3 = model3.fit(
    train_sentences,
    train_labels,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(3)
    ]
)

# Save the history to a file
with open('data/history3.pkl', 'wb') as file:
    pickle.dump(history3.history, file)
