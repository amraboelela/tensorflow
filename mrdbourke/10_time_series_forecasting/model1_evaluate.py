from model0_evaluate import *
from model1_init import *

model1.load_weights("data/" + model1.name)

print()
print("# Evaluate model on test data")

model1_evaluate = read_tensor("model1_evaluate")
if model1_evaluate is None:
    model1_evaluate = model1.evaluate(test_windows, test_labels)
    save_tensor(model1_evaluate, "model1_evaluate")
print(model1_evaluate)

print()
print("# Make predictions using model1 on the test dataset and view the results")
model1_preds = make_preds(model1, test_windows)
print(len(model1_preds), model1_preds[:10])

print()
print("# Evaluate preds")
model1_results = evaluate_preds(
    y_true=tf.squeeze(test_labels), # reduce to right shape
    y_pred=model1_preds
)
print(model1_results)

print()
print("# Naive results")
print(naive_results)

offset = 300
plt.figure(figsize=(10, 7))
# Account for the test_window offset and index into test_labels to ensure correct plotting
plot_time_series(
    timesteps=X_test[-len(test_windows):],
    values=test_labels[:, 0],
    start=offset,
    label="Test_data",
    name="time_series_model1"
)
plot_time_series(
    timesteps=X_test[-len(test_windows):],
    values=model1_preds,
    start=offset,
    format="-",
    label="model1_preds",
    name="time_series_model1"
)

print()
