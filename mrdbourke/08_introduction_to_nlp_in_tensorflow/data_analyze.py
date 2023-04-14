from common import *

print("train_df.head()")
print(str(train_df.head()))
print("")

print("train_df_shuffled.head()")
print(str(train_df_shuffled.head()))
print("")

# The test data doesn't have a target (that's what we'd try to predict)
print("test_df.head()")
print(str(test_df.head()))
print("")

# How many examples of each class?
print("train_df.target.value_counts()")
print(str(train_df.target.value_counts()))
print("")

# How many samples total?
print(f"Total training samples: {len(train_df)}")
print(f"Total test samples: {len(test_df)}")
print(f"Total samples: {len(train_df) + len(test_df)}")
print("")

random_index = random.randint(0, len(train_df)-5) # create random indexes not higher than the total number of samples
for row in train_df_shuffled[["text", "target"]][random_index:random_index+5].itertuples():
  _, text, target = row
  print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
  print(f"Text:\n{text}\n")
  print("---\n")
print("")

# Check the lengths
print("len(train_sentences), len(train_labels), len(val_sentences), len(val_labels)")
len(train_sentences), len(train_labels), len(val_sentences), len(val_labels)
print("")
