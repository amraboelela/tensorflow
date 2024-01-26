from model2_init import *

model2.load_weights("data/" + model2.name)

print()
print("# Evaluate model on test data")

model2_evaluate = read_tensor("model2_evaluate")
if model2_evaluate is None:
    model2_evaluate = model2.evaluate(test_windows, test_labels)
    save_tensor(model2_evaluate, "model2_evaluate")
print(model2_evaluate)

print()
print("# Get forecast predictions")
model2_preds = make_preds(
    model2,
    input_data=test_windows
)
  
print()
print("# Evaluate results for model 2 predictions")
model2_results = evaluate_preds(
    y_true=tf.squeeze(test_labels), # remove 1 dimension of test labels
    y_pred=model2_preds
)
print(model2_results)

offset = 300
plt.figure(figsize=(10, 7))
# Account for the test_window offset
plot_time_series(
    timesteps=X_test[-len(test_windows):],
    values=test_labels[:, 0],
    start=offset,
    label="test_data",
    name="time_series_model2"
)
plot_time_series(
    timesteps=X_test[-len(test_windows):],
    values=model2_preds,
    start=offset,
    format="-",
    label="model2_preds",
    name="time_series_model2"
)

print()
