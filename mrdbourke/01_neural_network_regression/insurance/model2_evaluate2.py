from common import *

model2 = load_model("data/model2.h5")

print("")
print("# Load the saved history2 object from a file")
with open('data/history2.pkl', 'rb') as f:
    history2 = pickle.load(f)
    
print("")
print("# Load the saved history2_2 object from a file")
with open('data/history2_2.pkl', 'rb') as f:
    history2_2 = pickle.load(f)
    
print("")
print("# Evaluate the model trained for 200 total epochs")
model2_loss, model2_mae = model2.evaluate(X_test_oh, y_test_oh)
print(model2_loss, model2_mae)

print("")
print("# Plot the model trained for 200 total epochs loss curves")
pd.DataFrame(history2_2).plot()
plt.ylabel("loss")
plt.xlabel("epochs") # note: epochs will only show 100 since we overrid the history variable
plt.savefig('data/images/loss2_2.png', format='png')

df2 = pd.DataFrame(history2)
df2_2 = pd.DataFrame(history2_2)

print("")
print("# history2")
print(df2)

print("")
print("# history2_2")
print(df2_2)

print("")
print("# Concatenate history2 and history2_2")
df2_2_c = pd.concat([df2, df2_2], ignore_index=True)
print(df2_2_c)

plt.figure()
df2_2_c.plot()
plt.ylabel("loss")
plt.xlabel("epochs") # note: epochs will only show 100 since we overrid the history variable
plt.savefig('data/images/loss2_2_c.png', format='png')

