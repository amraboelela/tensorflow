from model1_evaluate import *
from model4_init import *

print("")
print("# Get a summary of the model")
print(model4.summary())

model4.load_weights(checkpoint_path(4))

print("")
print("# Make predictions on the validation dataset")
model4_pred_probs = model4.predict(val_sentences)
print(model4_pred_probs.shape, model4_pred_probs[:10], "# view the first 10")

print("")
print("# Round out predictions and reduce to 1-dimensional array")
model4_preds = tf.squeeze(tf.round(model4_pred_probs))
print(model4_preds[:10])

print("")
print("# Calculate model results")
model4_results = calculate_results(
    y_true=val_labels,
    y_pred=model4_preds
)
print(model4_results)

print("")
print("# Compare model4 to model1")
compare_baseline_to_new_results(model1_results, model4_results)

print("")
