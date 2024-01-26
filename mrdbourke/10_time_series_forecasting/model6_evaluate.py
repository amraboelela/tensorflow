from model6_init import *

model6 = load_model("data/" + model6.name)

print()
print("# Evaluate model on test data")

model6_evaluate = read_tensor("model6_evaluate")
if model6_evaluate is None:
    model6_evaluate = model6.evaluate(X_test, y_test)
    save_tensor(model6_evaluate, "model6_evaluate")
print(model6_evaluate)

print()
print("# Make predictions on multivariate data")
model6_preds = tf.squeeze(model6.predict(X_test))
print(model6_preds[:10])

print()
print("# Evaluate preds")
model6_results = evaluate_preds(
    y_true=y_test,
    y_pred=model6_preds
)
print(model6_results)

print()
