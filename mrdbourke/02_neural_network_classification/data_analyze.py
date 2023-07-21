from common import *

print("# Check out the insurance dataset")
print(insurance.head())

print("")
print("# Turn all categories into numbers")
print(insurance_one_hot.head(), "# view the converted columns")

print("# View features")
print(X.head())

print(X_train_oh.head())
print(y_train_oh.head())

print("")
print("# Non-normalized and non-one-hot encoded data example")
print(X_train.loc[0])

print("")
print("# Normalized and one-hot encoded example")
print(X_train_normal[0])

print("")
print("# Notice the normalized/one-hot encoded shape is larger because of the extra columns")
print(X_train_normal.shape, X_train.shape)
