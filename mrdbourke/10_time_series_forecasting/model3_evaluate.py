from model0_evaluate import *
from model1_evaluate import *
from model2_evaluate import *
from model3_init import *

model3.load_weights("data/" + model3.name)

print()
print("# Evaluate model on test data")

model3_evaluate = read_tensor("model3_evaluate")
if model3_evaluate is None:
    model3_evaluate = model3.evaluate(test_windows, test_labels)
    save_tensor(model3_evaluate, "model3_evaluate")
print(model3_evaluate)

print()
print("# The predictions are going to be 7 steps at a time (this is the HORIZON size)")
model3_preds = make_preds(
    model3,
    input_data=test_windows
)
print(model3_preds[:5])

print()
print("# Get model3 results aggregated to single values")
model3_results = evaluate_preds(
    y_true=tf.squeeze(test_labels),
    y_pred=model3_preds
)
print(model3_results)

offset = 300
plt.figure(figsize=(10, 7))
plot_time_series(
    timesteps=X_test[-len(test_windows):],
    values=test_labels[:, 0],
    start=offset,
    label="Test_data",
    name="time_series_model3"
)

print()
print("# Checking the shape of model_3_preds results in [n_test_samples, HORIZON] (this will screw up the plot)")
plot_time_series(
    timesteps=X_test[-len(test_windows):],
    values=model3_preds,
    start=offset,
    label="model3_preds",
    name="time_series_model3"
)

offset = 300
plt.figure(figsize=(10, 7))
# Plot model_3_preds by aggregating them (note: this condenses information so the preds will look fruther ahead than the test data)
plot_time_series(
    timesteps=X_test[-len(test_windows):],
    values=test_labels[:, 0],
    start=offset,
    label="Test_data",
    name="time_series_model3_2"
)
plot_time_series(
    timesteps=X_test[-len(test_windows):],
    values=tf.reduce_mean(model3_preds, axis=1),
    format="-",
    start=offset,
    label="model3_preds",
    name="time_series_model3_2"
)

plt.figure()
pd.DataFrame( {
        "naive": naive_results["mae"],
        "horizon_1_window_7": model1_results["mae"],
        "horizon_1_window_30": model2_results["mae"],
        "horizon_7_window_30": model3_results["mae"]
    },
    index=["mae"]
).plot(figsize=(10, 7), kind="bar");
plt.savefig('data/images/model0_3.png', format='png')

print()
