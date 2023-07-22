from common2 import *

model3 = load_model("data/model3.keras")

model3.summary()

plot_model(model3, to_file='data/images/model3.png', show_shapes=True)

print("")
print("# Make predictions")
y_preds = model3.predict(X_test)
print("# View the predictions")
print(y_preds)

plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_preds,
                 index=3)

print("")
print("# Evaluate the model on the test set")
model3.evaluate(X_test, y_test)

print("")
print("# Calculate the mean absolute error")
mae = tf.metrics.mean_absolute_error(y_true=y_test,
                                     y_pred=y_preds)
print(mae)

print("")
print("# Check the test label tensor values")
print(y_test)

print("")
print("# Check the predictions tensor values (notice the extra square brackets)")
print(y_preds)

print("")
print("# Check the tensor shapes")
print(y_test.shape, y_preds.shape)

print("")
print("# Shape before squeeze()")
print(y_preds.shape)

print("")
print("# Shape after squeeze()")
print(y_preds.squeeze().shape)

print("")
print("# What do they look like?")
print(y_test, y_preds.squeeze())

print("")
print("# Calcuate the MAE")
mae = tf.metrics.mean_absolute_error(y_true=y_test,
                                     y_pred=y_preds.squeeze()) # use squeeze() to make same shape
print(mae)

print("")
print("# Calculate the MSE")
mse = tf.metrics.mean_squared_error(y_true=y_test,
                                    y_pred=y_preds.squeeze())
print(mse)

print("")
print("# Returns the same as tf.metrics.mean_absolute_error()")
print(tf.reduce_mean(tf.abs(y_test-y_preds.squeeze())))

print("")
print("# Calculate model3 metrics")
print(y_test, y_preds.squeeze())
                                        
mae3 = mean_absolute_error(y_test, y_preds.squeeze()).numpy()
mse3 = mean_squared_error(y_test, y_preds.squeeze()).numpy()
print(mae3, mse3)
