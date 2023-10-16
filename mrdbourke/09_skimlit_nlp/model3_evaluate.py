from model3_init import *

print()
print("# Get a summary of the model")
model3.summary()

model3.load_weights(checkpoint_path(3))

print()
print("# Evaluate on whole validation dataset")
model3.evaluate(valid_dataset)

print()
print("# Make predictions with feature extraction model")
model3_pred_probs = model3.predict(valid_dataset)
print(model3_pred_probs)

print()
print("# Convert the predictions with feature extraction model to classes")
model3_preds = tf.argmax(model3_pred_probs, axis=1)
print(model3_preds)

print()
print("# Calculate results from TF Hub pretrained embeddings results on validation set")
model3_results = calculate_results(
    y_true=val_labels_encoded,
    y_pred=model3_preds
)
print(model3_results)

print()

