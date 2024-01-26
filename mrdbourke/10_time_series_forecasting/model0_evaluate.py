from model0_init import *

print(naive_forecast[:10], naive_forecast[-10:]) # View frist 10 and last 10

print()
print("# Plot naive forecast")
plt.figure(figsize=(10, 7))
plot_time_series(
    timesteps=X_train,
    values=y_train,
    label="Train data",
    name="time_series_naive"
)
plot_time_series(
    timesteps=X_test,
    values=y_test,
    label="Test data",
    name="time_series_naive"
)
plot_time_series(
    timesteps=X_test[1:],
    values=naive_forecast,
    format="-",
    label="Naive forecast",
    name="time_series_naive"
)

plt.figure(figsize=(10, 7))
offset = 300 # offset the values by 300 timesteps
plot_time_series(
    timesteps=X_test,
    values=y_test,
    start=offset,
    label="Test data",
    name="time_series_naive_300"
)
plot_time_series(
    timesteps=X_test[1:],
    values=naive_forecast,
    format="-",
    start=offset,
    label="Naive forecast",
    name="time_series_naive_300"
)

print()
naive_results = evaluate_preds(
    y_true=y_test[1:],
    y_pred=naive_forecast
)

print(naive_results)

print()
