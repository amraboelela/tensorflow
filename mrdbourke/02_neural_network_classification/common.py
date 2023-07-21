import sys
sys.path.append('../../modules')
from helper_functions import *

insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
                                                                
# Turn all categories into numbers
insurance_one_hot = pd.get_dummies(insurance, dtype=int)

# Create X_oh & y_oh values
X_oh = insurance_one_hot.drop("charges", axis=1)
y_oh = insurance_one_hot["charges"]

# Create training and test sets
X_train_oh, X_test_oh, y_train_oh, y_test_oh = train_test_split(X_oh,
                                                                y_oh,
                                                                test_size=0.2,
                                                                random_state=42) # set random state for reproducible splits

# Create X & y values
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
                                                    
# Create column transformer (this will help us normalize/preprocess our data)
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]), # get all values between 0 and 1
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

# Fit column transformer on the training data only (doing so on test data would result in data leakage)"
ct.fit(X_train)

# Transform training and test data with normalization (MinMaxScalar) and one hot encoding (OneHotEncoder)"
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)
