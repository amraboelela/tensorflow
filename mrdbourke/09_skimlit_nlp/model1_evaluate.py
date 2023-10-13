from model1_fit import *

print()
print("# Evaluate baseline on validation dataset")
model1_score = model1.score(
    X=val_sentences,
    y=val_labels_encoded
)
print(model1_score)

print()
print("# Make predictions")
model1_preds = model1.predict(val_sentences)
print(model1_preds)

print()
print("# Calculate baseline results")
model1_results = calculate_results(
    y_true=val_labels_encoded,
    y_pred=model1_preds
)
print(model1_results)

