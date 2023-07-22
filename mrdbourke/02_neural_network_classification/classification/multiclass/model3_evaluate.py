from model3_init import *

model3 = load_model("data/model3.keras")

print("")
print("# Load the saved history object from a file")
with open('data/history3.pkl', 'rb') as f:
    history3 = pickle.load(f)

print("")
print("# Plot the learning rate decay curve")
lrs = 1e-3 * (10**(np.arange(40)/20))
plt.semilogx(lrs, history3["loss"]) # want the x-axis to be log-scale
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.title("Finding the ideal learning rate");
plt.savefig('data/images/learning_rate.png', format='png')
