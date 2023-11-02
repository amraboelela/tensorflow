from model6_init import *

print()
print("# Get a summary of our token, char and positional embedding model")
model6.summary()

print()
print("# Plot the token, char, positional embedding model")
plot_model(model6, to_file='data/images/plot_model6.png')

exit()

model5.load_weights(checkpoint_path(5))

print()
print("# Plot hybrid token and character model")
plot_model(model5, to_file='data/images/plot_model5.png')
    
print()
print("# Evaluate on the whole validation dataset")
model5.evaluate(val_char_token_dataset)

print()
print("# Make predictions using the token-character model hybrid")
model5_pred_probs = model5.predict(val_char_token_dataset)
print(model5_pred_probs)

print()
print("# Turn prediction probabilities into prediction classes")
model5_preds = tf.argmax(model5_pred_probs, axis=1)
print(model5_preds)

print()
print("# Get results of token-char-hybrid model")
model5_results = calculate_results(
    y_true=val_labels_encoded,
    y_pred=model5_preds
)
print(model5_results)

print()

