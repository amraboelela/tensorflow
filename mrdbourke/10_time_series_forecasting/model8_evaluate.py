from model8_init import *

print()
print("# Get list of trained ensemble models")
ensemble_models = get_ensemble_models(
    num_iter=5,
    num_epochs=1000
)

print()
print("# Create a list of ensemble predictions")
ensemble_preds = make_ensemble_preds(
    ensemble_models=ensemble_models,
    data=test_dataset
)
print(ensemble_preds)

print()
print("# Evaluate ensemble model(s) predictions")
ensemble_results = evaluate_preds(
    y_true=y_test,
    y_pred=np.median(ensemble_preds, axis=0)
) # take the median across all ensemble predictions
print(ensemble_results)

print()
print("# Get the upper and lower bounds of the 95%")
lower, upper = get_upper_lower(preds=ensemble_preds)

print()
print("# Get the median values of our ensemble preds")
ensemble_median = np.median(ensemble_preds, axis=0)

print()
print("# Plot the median of our ensemble preds along with the prediction intervals (where the predictions fall between)")
offset=500
plt.figure(figsize=(10, 7))
plt.plot(X_test.index[offset:], y_test[offset:], "g", label="Test Data")
plt.plot(X_test.index[offset:], ensemble_median[offset:], "k-", label="Ensemble Median")
plt.xlabel("Date")
plt.ylabel("BTC Price")
plt.fill_between(
    X_test.index[offset:],
    (lower)[offset:],
    (upper)[offset:], label="Prediction Intervals"
)
plt.legend(loc="upper left", fontsize=14);
plt.savefig('data/images/Prediction_Intervals.png', format='png')
