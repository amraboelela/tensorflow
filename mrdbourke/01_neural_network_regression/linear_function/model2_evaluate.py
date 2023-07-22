from common1 import *

model2 = load_model("data/model2.keras")

print("")
print("# Remind ourselves of what X and y are")
print(X, y)

print("")
print("# Try and predict what y would be if X was 17.0")
print(model2.predict([17.0]), "# the right answer is 27.0 (y = X + 10)")
