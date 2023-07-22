from model2_init import *

model2 = load_model("data/model2.keras")

print("")
print("# Load the saved history object from a file")
with open('data/history2.pkl', 'rb') as f:
    history2 = pickle.load(f)

print("")
print("# Plot normalized data loss curves")
pd.DataFrame(history2).plot(title="Normalized data");
plt.savefig('data/images/history2.png', format='png')
