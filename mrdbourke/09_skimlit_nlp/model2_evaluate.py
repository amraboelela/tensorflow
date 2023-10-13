from model2_init import *

print()
print("# Get summary of Conv1D model")
model2.summary()

model2.load_weights(checkpoint_path(2))

print()
print("# Evaluate on whole validation dataset (we only validated on 10% of batches during training)")
model2.evaluate(valid_dataset)

print()
print("# Make predictions (our model outputs prediction probabilities for each class)")
model2_pred_probs = model2.predict(valid_dataset)
print(model2_pred_probs)

print()
print("# Convert pred probs to classes")
model2_preds = tf.argmax(model2_pred_probs, axis=1)
print(model2_preds)

print()


