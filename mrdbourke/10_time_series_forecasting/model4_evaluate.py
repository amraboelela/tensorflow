from model4_init import *

model4 = load_model("data/" + model4.name)

model4.summary()

print()
print("# Evaluate model on test data")

model4_evaluate = read_tensor("model4_evaluate")
if model4_evaluate is None:
    model4_evaluate = model4.evaluate(test_windows, test_labels)
    save_tensor(model4_evaluate, "model4_evaluate")
print(model4_evaluate)

print()
print("# Make predictions")
model4_preds = make_preds(model4, test_windows)
model4_preds[:10]

print()
print("# Evaluate predictions")
model4_results = evaluate_preds(
    y_true=tf.squeeze(test_labels),
    y_pred=model4_preds
)
print(model4_results)

print()
