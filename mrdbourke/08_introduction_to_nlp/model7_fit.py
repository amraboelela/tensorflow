from model7_init import *
   
tensorboard_path = "NLP/model7"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Train a classifier on top of pretrained embeddings
history7 = model7.fit(
    train_sentences,
    train_labels,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(7)
    ]
)

# Save the history to a file
with open('data/history7.pkl', 'wb') as file:
    pickle.dump(history7.history, file)
