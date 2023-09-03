from model2_init import *
   
tensorboard_path = "NLP/model2"
subprocess.run(['rm', '-r', "data/" + tensorboard_path])

# Fit the model
history2 = model2.fit(
    train_sentences, # input sentences can be a list of strings due to text preprocessing layer built-in model
    train_labels,
    epochs=5,
    validation_data=(val_sentences, val_labels),
    callbacks=[
        tensorboard_callback(tensorboard_path),
        checkpoint_callback(2)
    ]
)

# Save the history to a file
with open('data/history2.pkl', 'wb') as file:
    pickle.dump(history2.history, file)
