from model6_evaluate import *
from model7_init import *

print()
print("# Make predictions with the loaded model on the validation set")
loaded_pred_probs = loaded_model.predict(val_pos_char_token_dataset, verbose=1)
loaded_preds = tf.argmax(loaded_pred_probs, axis=1)
print(loaded_preds[:10])

print()
print("# Evaluate loaded model's predictions")
loaded_model_results = calculate_results(
    val_labels_encoded,
    loaded_preds
)
print(loaded_model_results)

print()
print("# Compare loaded model results with original trained model results (should be quite close)")
print(np.isclose(list(model6_results.values()), list(loaded_model_results.values()), rtol=1e-02))

print()
print("# Check loaded model summary (note the number of trainable parameters)")
print(loaded_model.summary())

print()
print("# Check shapes")
print(test_pos_char_token_dataset)

print()
print("# Make predictions on the test dataset")
test_pred_probs = loaded_model.predict(
    test_pos_char_token_dataset,
    verbose=1
)
test_preds = tf.argmax(test_pred_probs, axis=1)
print(test_preds[:10])

print()
print("# Evaluate loaded model test predictions")
loaded_model_test_results = calculate_results(
    y_true=test_labels_encoded,
    y_pred=test_preds
)
print(loaded_model_test_results)

print()
print("# Get list of class names of test predictions")
test_pred_classes = [label_encoder.classes_[pred] for pred in test_preds]
print(test_pred_classes)

print()
print("# Create prediction-enriched test dataframe")
test_df["prediction"] = test_pred_classes # create column with test prediction class names
test_df["pred_prob"] = tf.reduce_max(test_pred_probs, axis=1).numpy() # get the maximum prediction probability
test_df["correct"] = test_df["prediction"] == test_df["target"] # create binary column for whether the prediction is right or not
print(test_df.head(20))

print()
print("# Find top 100 most wrong samples (note: 100 is an abitrary number, you could go through all of them if you wanted)")
top_100_wrong = test_df[test_df["correct"] == False].sort_values("pred_prob", ascending=False)[:100]
print(top_100_wrong)

print()
print("# Investigate top wrong preds")
for row in top_100_wrong[0:10].itertuples(): # adjust indexes to view different samples
  _, target, text, line_number, total_lines, prediction, pred_prob, _ = row
  print(f"Target: {target}, Pred: {prediction}, Prob: {pred_prob}, Line number: {line_number}, Total lines: {total_lines}\n")
  print(f"Text:\n{text}\n")
  print("-----\n")
  
print()
