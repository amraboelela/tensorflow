from common import *
from model2_evaluate2 import *

model3 = load_model("data/model3.h5")

print("")
print("# Load the saved history object from a file")
with open('data/history3.pkl', 'rb') as f:
    history3 = pickle.load(f)
    
print("")
print("# Evaulate 3rd model")
model3_loss, model3_mae = model3.evaluate(X_test_normal, y_test)

print("")
print("# Compare modelling results from non-normalized data and normalized data")
print(model2_mae, model3_mae)

# Plot history (also known as a loss curve)
pd.DataFrame(history3).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.savefig('data/images/loss3.png', format='png')
