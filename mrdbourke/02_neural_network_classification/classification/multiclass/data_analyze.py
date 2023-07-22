from common import *

print("")
print("# Show the first training example")
print(f"Training sample:\n{train_data[0]}\n")
print(f"Training label: {train_labels[0]}")

print("")
print("# Check the shape of our data")
print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)

print("")
print("# Check shape of a single example")
print(train_data[0].shape, train_labels[0].shape)

print("")
print("# Plot a single example")
plt.imshow(train_data[7])
plt.savefig('data/images/train_data7.png', format='png')

print("")
print("# Check our samples label")
print(train_labels[7])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("")
print("# How many classes are there (this'll be our output shape)?")
print(len(class_names))

print("")
print("# Plot an example image and its label")
plt.figure()
plt.imshow(train_data[17], cmap=plt.cm.binary) # change the colours to black & white
plt.title(class_names[train_labels[17]])
plt.savefig('data/images/train_data17.png', format='png')
