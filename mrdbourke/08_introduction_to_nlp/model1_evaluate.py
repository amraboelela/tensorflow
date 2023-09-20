from model1_fit import *

print("")
baseline_score = model1.score(val_sentences, val_labels)
print(f"Our baseline model achieves an accuracy of: {baseline_score*100:.2f}%")

print("")
print("# Make predictions")
model1_preds = model1.predict(val_sentences)
print(model1_preds[:20])

print("")
print("# Get baseline results")
model1_results = calculate_results(
    y_true=val_labels,
    y_pred=model1_preds
)
print(model1_results)

