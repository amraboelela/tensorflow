from model5_init import *

model5 = load_model("data/" + model5.name)

model5.summary()

print()
print("# Evaluate model on test data")

model5_evaluate = read_tensor("model5_evaluate")
if model5_evaluate is None:
    model5_evaluate = model5.evaluate(test_windows, test_labels)
    save_tensor(model5_evaluate, "model5_evaluate")
print(model5_evaluate)

print()
print("# Make predictions")
model5_preds = make_preds(model5, test_windows)
model5_preds[:10]

print()
print("# Evaluate predictions")
model5_results = evaluate_preds(
    y_true=tf.squeeze(test_labels),
    y_pred=model5_preds
)
print(model5_results)

print()
