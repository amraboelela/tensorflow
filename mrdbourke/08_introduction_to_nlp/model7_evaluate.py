from model1_evaluate import *
from model7_init import *

print("")
print("# Get a summary of the model")
print(model7.summary())

model7.load_weights(checkpoint_path(7))

print("")
print("# Make predictions on the validation dataset")
model7_pred_probs = model7.predict(val_sentences)
print(model7_pred_probs.shape, model7_pred_probs[:10], "# view the first 10")

print("")
print("# Round out predictions and reduce to 1-dimensional array")
model7_preds = tf.squeeze(tf.round(model7_pred_probs))
print(model7_preds[:10])

print("")
print("# Calculate model results")
model7_results = calculate_results(
    y_true=val_labels,
    y_pred=model7_preds
)
print(model7_results)

print("")
print("# Compare model7 to baseline")
compare_baseline_to_new_results(model1_results, model7_results)

print("")
