from common import *

print("")
print("# Read in the insurance dataset")
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

print("# Check out the insurance dataset")
print(insurance.head())

print("")
print("# Turn all categories into numbers")
insurance_one_hot = pd.get_dummies(insurance, dtype=int)
print(insurance_one_hot.head(), "# view the converted columns")

print("")
print("# Create X & y values")
X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]

print("# View features")
print(X.head())

print("")
print("# Create training and test sets")
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42) # set random state for reproducible splits
                                                    
print(X_train.head())
print(y_train.head())
