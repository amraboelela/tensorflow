from model2_init import *

model2 = load_model("data/model2.keras")

# Try training for a little longer (100 more epochs)
history2_2 = model2.fit(X_train_oh, y_train_oh, epochs=100, verbose=0)

model2.save('data/model2.keras')

# Save the history to a file
with open('data/history2_2.pkl', 'wb') as file:
    pickle.dump(history2_2.history, file)
