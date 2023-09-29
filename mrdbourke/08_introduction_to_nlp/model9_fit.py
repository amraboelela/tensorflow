from model9_init import *

tensorboard_path = "NLP/model9"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Train a classifier on top of pretrained embeddings
history9 = model9.fit(
    train_sentences,
    train_labels,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(9)
    ]
)

# Save the history to a file
with open('data/history9.pkl', 'wb') as file:
    pickle.dump(history9.history, file)
