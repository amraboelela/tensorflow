from model2_init import *

model2 = load_model("data/model2.keras")

print("")
print("# Evaluate the model")
model2.evaluate(X, y)
