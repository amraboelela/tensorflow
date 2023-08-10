from common import *

model2 = load_model("data/model2.keras")

# Load the saved history object from a file
with open('data/history2.pkl', 'rb') as f:
    history2 = pickle.load(f)

print("")
print("# Check the results of the insurance model")
model2_evaluate = read_tensor("model2_evaluate")
if model2_evaluate is None:
    model2_evaluate = model2.evaluate(X_test_oh, y_test_oh)
    save_tensor(model2_evaluate, "model2_evaluate")
print(model2_evaluate)

# Plot history (also known as a loss curve)
pd.DataFrame(history2).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.savefig('data/images/loss2.png', format='png')
