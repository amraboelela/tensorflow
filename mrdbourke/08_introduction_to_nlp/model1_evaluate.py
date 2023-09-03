from model1_fit import *

baseline_score = model1.score(val_sentences, val_labels)
print(f"Our baseline model achieves an accuracy of: {baseline_score*100:.2f}%")

print("")
print("# Make predictions")
baseline_preds = model1.predict(val_sentences)
print(baseline_preds[:20])

print("")
print("# Get baseline results")
baseline_results = calculate_results(
    y_true=val_labels,
    y_pred=baseline_preds
)
print(baseline_results)

