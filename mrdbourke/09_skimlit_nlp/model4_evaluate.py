from model4_init import *

print()
print("# Get a summary of the model")
model4.summary()

model4.load_weights(checkpoint_path(4))

print()
print("# Evaluate model4 on whole validation char dataset")
model4.evaluate(val_char_dataset)

print()
print("# Make predictions with character model only")
model4_pred_probs = model4.predict(val_char_dataset)
print(model4_pred_probs)

print()
print("# Convert predictions to classes")
model4_preds = tf.argmax(model4_pred_probs, axis=1)
print(model4_preds)

print()
print("# Calculate Conv1D char only model results")
model4_results = calculate_results(
    y_true=val_labels_encoded,
    y_pred=model4_preds
)
print(model4_results)

print()

