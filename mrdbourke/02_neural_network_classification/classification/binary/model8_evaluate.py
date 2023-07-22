from model8_init import *

model8 = load_model("data/model8.keras")

print("")
print("# Load the saved history object from a file")
with open('data/history8.pkl', 'rb') as f:
    history8 = pickle.load(f)

# Evaluate our model on the test set
loss, accuracy = model8.evaluate(X_test, y_test)
print(f"Model loss on the test set: {loss}")
print(f"Model accuracy on the test set: {100*accuracy:.2f}%")

print("")
print("# Check the deicison boundary (blue is blue class, yellow is the crossover, red is red class)")
plot_decision_boundary(model8, X, y, 8)

print("")
print("# Plot the decision boundaries for the training and test sets")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model8, X=X_train, y=y_train, index=8)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model8, X=X_test, y=y_test, index=8)

print("")
print("# You can access the information in the history variable using the .history attribute")
print(pd.DataFrame(history8))

print("")
print("# Plot the loss curves")
plt.figure()
pd.DataFrame(history8).plot()
plt.title("Model8 training curves")
plt.savefig('data/images/history8.png', format='png')
    
