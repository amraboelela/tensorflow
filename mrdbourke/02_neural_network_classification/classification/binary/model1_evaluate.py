from model1_init import *

model1 = load_model("data/model1.keras")

print("")
print("# Train our model for longer (more chances to look at the data)")
model1.fit(X, y, epochs=200, verbose=0) # set verbose=0 to remove training updates

model1_evaluate = read_tensor("model1_evaluate")
if model1_evaluate is None:
    model1_evaluate = model1.evaluate(X, y)
    save_tensor(model1_evaluate, "model1_evaluate")
print(model1_evaluate)
