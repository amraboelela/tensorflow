from common import *

model1 = load_model("data/model1.keras")

print("")
print("# Check the results of the insurance model")
model1_evaluate = read_tensor("model1_evaluate")
if model1_evaluate is None:
    model1_evaluate = model1.evaluate(X_test_oh, y_test_oh)
    save_tensor(model1_evaluate, "model1_evaluate")
print(model1_evaluate)
