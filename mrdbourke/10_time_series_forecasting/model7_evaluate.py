from model7_init import *

print()
print("# Make train and test sets")
print(len(X_train), len(y_train), len(X_test), len(y_test))

print()
print("# 3. Batch and prefetch for optimal performance")
print(train_dataset, test_dataset)

print()
print("# Values from N-BEATS paper Figure 1 and Table 18/Appendix D")
print(INPUT_SIZE, THETA_SIZE)

model7 = load_model("data/" + model7.name)

print()
print("# Evaluate N-BEATS model on the test dataset")
model7_evaluate = read_tensor("model7_evaluate")
if model7_evaluate is None:
    model7_evaluate = model7.evaluate(X_test, y_test)
    save_tensor(model7_evaluate, "model7_evaluate")
print(model7_evaluate)

print()
print("# Make predictions with N-BEATS model")
model7_preds = make_preds(model7, test_dataset)
print(model7_preds[:10])

print()
print("# Evaluate N-BEATS model predictions")
model7_results = evaluate_preds(
    y_true=y_test,
    y_pred=model7_preds
)
print(model7_results)

print()
print("# Plot the N-BEATS model and inspect the architecture")
plot_model(model7, to_file='data/images/plot_model7.png')

print()
