from model2_init import *

model2 = load_model("data/model2.keras")

print("")
print("# Evaluate the model")
model2_evaluate = read_tensor("model2_evaluate")
if model2_evaluate is None:
    model2_evaluate = model2.evaluate(X, y)
    save_tensor(model2_evaluate, "model2_evaluate")
print(model2_evaluate)
