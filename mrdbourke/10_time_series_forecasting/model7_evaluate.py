from model7_init import *

print()
print("# Make train and test sets")
print(len(X_train), len(y_train), len(X_test), len(y_test))

print()
print("# 3. Batch and prefetch for optimal performance")
print(train_dataset, test_dataset)

print()
print("# Values from N-BEATS paper Figure 1 and Table 18/Appendix D")
print(INPUT_SIZE, THETA_SIZE)

print()
