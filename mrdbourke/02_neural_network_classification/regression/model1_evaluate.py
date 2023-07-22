from model1_init import *

model1 = load_model("data/model1.keras")

# Make predictions with our trained model
y_reg_preds = model1.predict(y_reg_test)

# Plot the model's predictions against our regression data
plt.figure(figsize=(10, 7))
plt.scatter(X_reg_train, y_reg_train, c='b', label='Training data')
plt.scatter(X_reg_test, y_reg_test, c='g', label='Testing data')
plt.scatter(X_reg_test, y_reg_preds.squeeze(), c='r', label='Predictions')
plt.legend()
plt.savefig('data/images/predict.png', format='png')
