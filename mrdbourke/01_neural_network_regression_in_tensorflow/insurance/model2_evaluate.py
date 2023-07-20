from common import *

model2 = load_model("data/model2.h5")

# Load the saved history object from a file
with open('data/history2.pkl', 'rb') as f:
    history2 = pickle.load(f)
    
# Check the results of the insurance model
model2.evaluate(X_test, y_test)

# Plot history (also known as a loss curve)
pd.DataFrame(history2).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.savefig('data/images/loss2.png', format='png')
