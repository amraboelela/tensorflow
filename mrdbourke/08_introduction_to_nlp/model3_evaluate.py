from model1_evaluate import *
from model3_init import *

print("")
print("# Get a summary of the model")
print(model3.summary())

model3.load_weights(checkpoint_path(3))

print("")
print("# Make predictions on the validation dataset")
model3_pred_probs = model3.predict(val_sentences)
print(model3_pred_probs.shape, model3_pred_probs[:10], "# view the first 10")

print("")
print("# Round out predictions and reduce to 1-dimensional array")
model3_preds = tf.squeeze(tf.round(model3_pred_probs))
print(model3_preds[:10])

print("")
print("# Calculate model results")
model3_results = calculate_results(
    y_true=val_labels,
    y_pred=model3_preds
)
print(model3_results)

print("")
print("# Compare model3 to baseline")
compare_baseline_to_new_results(model1_results, model3_results)

print("")
