from model1_evaluate import *
from model5_init import *

print("")
print("# Get a summary of the model")
print(model5.summary())

model5.load_weights(checkpoint_path(5))

print("")
print("# Make predictions on the validation dataset")
model5_pred_probs = model5.predict(val_sentences)
print(model5_pred_probs.shape, model5_pred_probs[:10], "# view the first 10")

print("")
print("# Round out predictions and reduce to 1-dimensional array")
model5_preds = tf.squeeze(tf.round(model5_pred_probs))
print(model5_preds[:10])

print("")
print("# Calculate model results")
model5_results = calculate_results(
    y_true=val_labels,
    y_pred=model5_preds
)
print(model5_results)

print("")
print("# Compare model5 to baseline")
compare_baseline_to_new_results(model1_results, model5_results)

print("")
