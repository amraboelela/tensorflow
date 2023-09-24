from model1_evaluate import *
from model7_evaluate import *
from model9_init import *

print("")
print("# Get a summary of the model")
print(model9.summary())

model9.load_weights(checkpoint_path(9))

print("")
print("# Make predictions on the validation dataset")
model9_pred_probs = model9.predict(val_sentences)
print(model9_pred_probs.shape, model9_pred_probs[:10], "# view the first 10")

print("")
print("# Round out predictions and reduce to 1-dimensional array")
model9_preds = tf.squeeze(tf.round(model9_pred_probs))
print(model9_preds[:10])

print("")
print("# Calculate model results")
model9_results = calculate_results(
    y_true=val_labels,
    y_pred=model9_preds
)
print(model9_results)

print("")
print("# Compare model9 to model1")
compare_baseline_to_new_results(model1_results, model9_results)

print("")
print("# Compare model9 to model7")
compare_baseline_to_new_results(model7_results, model9_results)
print("")
