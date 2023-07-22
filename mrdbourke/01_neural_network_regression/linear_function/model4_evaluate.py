from common2 import *

model4 = load_model("data/model4.keras")

model4.summary()

plot_model(model4, to_file='data/images/model4.png', show_shapes=True)

print("")
print("# Make and plot predictions for model4")
y_preds4 = model4.predict(X_test)
plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_preds4,
                 index=4)

print("")
print("# Calculate model4 metrics")
mae4 = mean_absolute_error(y_test, y_preds4.squeeze()).numpy()
mse4 = mean_squared_error(y_test, y_preds4.squeeze()).numpy()
print(mae4, mse4)
