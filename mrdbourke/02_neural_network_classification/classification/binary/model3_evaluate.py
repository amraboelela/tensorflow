from model3_init import *

model3 = load_model("data/model3.keras")

print("")
print("# Check out the predictions our model is making")
plot_decision_boundary(model3, X, y, 3)
            
print(model3.summary())
