from model1_init import *

model1 = load_model("data/model1.keras")

print("")
print("# Load the saved history object from a file")
with open('data/history1.pkl', 'rb') as f:
    history1 = pickle.load(f)
    
print("")
print("# Plot non-normalized data loss curves")
pd.DataFrame(history1).plot(title="Non-normalized Data")
plt.savefig('data/images/history1.png', format='png')

print("")
print("# Check the shapes of our model")
print("# Note: the 'None' in (None, 784) is for batch_size, we'll cover this in a later module")
model1.summary()

print("")
print("# Check the min and max values of the training data")
print(train_data.min(), train_data.max())

print("")
print("# Divide train and test images by the maximum value (normalize it)")
train_data = train_data / 255.0
test_data = test_data / 255.0

print("# Check the min and max values of the training data")
print(train_data.min(), train_data.max())

