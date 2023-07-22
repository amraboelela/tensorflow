from model9_init import *

model9 = load_model("data/model9.keras")

print("")
print("# Load the saved history object from a file")
with open('data/history9.pkl', 'rb') as f:
    history9 = pickle.load(f)
    
print("")
print("# Checkout the history")
plt.figure()
pd.DataFrame(history9).plot(figsize=(10,7), xlabel="epochs");
plt.savefig('data/images/history9.png', format='png')

print("")
print("# Plot the learning rate versus the loss")
lrs = 1e-4 * (10 ** (np.arange(100)/20))
plt.figure()
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history9["loss"]) # we want the x-axis (learning rate) to be log scale
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs. loss");
plt.savefig('data/images/learning_rate.png', format='png')

print("")
print("# Example of other typical learning rate values")
print(10**0, 10**-1, 10**-2, 10**-3, 1e-4)
