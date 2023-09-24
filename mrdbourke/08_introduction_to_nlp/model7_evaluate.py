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
print("# Compare model7 to model1")
compare_baseline_to_new_results(model1_results, model7_results)

print("")

#print("# Load model with custom Hub Layer (required with HDF5 format)")
#loaded_model7 = tf.keras.models.load_model(
#    "data/model7.h5",
#    custom_objects={"KerasLayer": hub.KerasLayer}
#)
                                            
#print("# How does our loaded model perform?")
#loaded_model7.evaluate(val_sentences, val_labels)
#print("")

#print("# Load TF Hub Sentence Encoder SavedModel")
#loaded_model7_SavedModel = tf.keras.models.load_model("model7_SavedModel_format")

print("")
print("# Evaluate loaded SavedModel format")
model7_evaluate = read_tensor("model7_evaluate")
if model7_evaluate is None:
    model7_evaluate = model7.evaluate(val_sentences, val_labels)
    save_tensor(model7_evaluate, "model7_evaluate")
print(model7_evaluate)

print("# Create dataframe with validation sentences and best performing model predictions")
val_df = pd.DataFrame({
    "text": val_sentences,
    "target": val_labels,
    "pred": model7_preds,
    "pred_prob": tf.squeeze(model7_pred_probs)
})
print(val_df.head())
print("")

print("# Find the wrong predictions and sort by prediction probabilities")
most_wrong = val_df[val_df["target"] != val_df["pred"]].sort_values("pred_prob", ascending=False)
print(most_wrong[:10])
print("")

print("# Check the false positives (model predicted 1 when should've been 0)")
for row in most_wrong[:10].itertuples(): # loop through the top 10 rows (change the index to view different rows)
  _, text, target, pred, prob = row
  print(f"Target: {target}, Pred: {int(pred)}, Prob: {prob}")
  print(f"Text:\n{text}\n")
  print("----\n")
print("")

print("# Check the most wrong false negatives (model predicted 0 when should've predict 1)")
for row in most_wrong[-10:].itertuples():
  _, text, target, pred, prob = row
  print(f"Target: {target}, Pred: {int(pred)}, Prob: {prob}")
  print(f"Text:\n{text}\n")
  print("----\n")
print("")
