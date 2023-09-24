from model1_evaluate import *
from model6_init import *

print("")
print("# Get a summary of the model")
print(model6.summary())

model6.load_weights(checkpoint_path(6))

print("")
print("# Make predictions on the validation dataset")
model6_pred_probs = model6.predict(val_sentences)
print(model6_pred_probs.shape, model6_pred_probs[:10], "# view the first 10")

print("")
print("# Round out predictions and reduce to 1-dimensional array")
model6_preds = tf.squeeze(tf.round(model6_pred_probs))
print(model6_preds[:10])

print("")
print("# Calculate model results")
model6_results = calculate_results(
    y_true=val_labels,
    y_pred=model6_preds
)
print(model6_results)

print("")
print("# Compare model6 to model1")
compare_baseline_to_new_results(model1_results, model6_results)

print("")
