from common import *

print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

print(filenames)

print()
print("# the whole first example of an abstract + a little more of the next one")
print(train_lines[:20])

print(len(train_samples), len(val_samples), len(test_samples))

print()
print("# Check the first abstract of our training data")
print(train_samples[:14])

print()
train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)
print(train_df.head(14))

print()
print("# Distribution of labels in training data")
print(train_df.target.value_counts())

train_df.total_lines.plot.hist()
plt.savefig('data/images/train_df.png', format='png')
