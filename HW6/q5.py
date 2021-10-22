import pandas as pd
import numpy as np
from sklearn import linear_model

np.set_printoptions(edgeitems=5)

admission_data = pd.read_csv('./datasets/q5.csv')
# First 400 rows - Training data
training_data = admission_data[:400]
# Last 100 rows - Test data
test_data = admission_data[-100:]

# number of training data observations
n_train = training_data.shape[0]
# number of test data observations
n_test = test_data.shape[0]

# Dependant Matrix Y of dimension n * 1, which is the set of observed Admit Chance in the training data
Y_train = training_data.iloc[:, :1].values
# Dependant Matrix Y of dimension n * 1, which is the set of observed Admit Chance in the test data
Y_test_actual = test_data.iloc[:, :1].values

print('----------------------------------------- PART a) --------------------------------------')
# Step 1) Calculate the Beta matrix using the training data observations
# number of features
k = training_data.shape[1] - 1
# Feature matrix X of dimension n * k (does not include the intercept beta_0) on the training data
X_train = training_data.iloc[:, 1:].values
X_train_T = X_train.transpose()

# To find - the Beta coefficient matrix of dimension k * 1 (does not include the intercept beta_0), of the form beta_1 to beta_k
# The Beta matrix is given by beta_hat = inverse(X_T * X) * (X_T * Y)
inverse_term = np.linalg.inv(X_train_T.dot(X_train))
beta_hat = inverse_term.dot(X_train_T.dot(Y_train))
print(pd.DataFrame(beta_hat, columns=['Beta Values'], index=['β'+ str(i) for i in range(1, k+1)]))

# Step 2) Using the Beta matrix obtained, calculate the predicted values for Chance of Admit on the test data
# Feature matrix X of dimension n * k (does not include the intercept beta_0) on the test data
X_test = test_data.iloc[:, 1:].values
# Calculate Prediction Matrix Y_hat = X_test * beta_cap, which has dimension n * 1
Y_hat = X_test.dot(beta_hat).round(3)
# print(Y_hat.flatten())

# Calculate SSE
residual_cap = Y_test_actual - Y_hat
sse = (residual_cap ** 2).sum()
print("SSE using all 7 features is {0}\n".format(round(sse, 4)))

## Using sklearn's LinearRegression without fitting the intercept beta_0, for comparison with the result we got above. Both should ideally be the same.
# mlr_model = linear_model.LinearRegression(fit_intercept=False)
# mlr_model.fit(X_train, Y_train)
# Y_pred = mlr_model.predict(X_test)
# print(pd.DataFrame(mlr_model.coef_.transpose(), columns=['sklearn Beta Values'], index=['β'+ str(i) for i in range(1, k+1)]))
# print("SSE using sklearn with all 7 features is {0}\n".format(round(((Y_test_actual - Y_pred) ** 2).sum(), 4)))

print('----------------------------------------- PART b) --------------------------------------')
k = 3
X_train = training_data[['TOEFL Score', 'SOP', 'LOR']].values
X_train_T = X_train.transpose()

inverse_term = np.linalg.inv(X_train_T.dot(X_train))
beta_hat = inverse_term.dot(X_train_T.dot(Y_train))
print(pd.DataFrame(beta_hat, columns=['Beta Values'], index=['β'+ str(i) for i in range(1, k+1)]))

X_test = test_data[['TOEFL Score', 'SOP', 'LOR']].values
Y_hat = X_test.dot(beta_hat).round(3)
# print(Y_hat.flatten())

# Calculate SSE
residual_cap = Y_test_actual - Y_hat
sse = (residual_cap ** 2).sum()
print("SSE using TOEFL, SOP and LOR features is {0}\n".format(round(sse, 4)))

print('----------------------------------------- PART c) --------------------------------------')
k = 2
X_train = training_data[['GRE Score', 'GPA']].values
X_train_T = X_train.transpose()

inverse_term = np.linalg.inv(X_train_T.dot(X_train))
beta_hat = inverse_term.dot(X_train_T.dot(Y_train))
print(pd.DataFrame(beta_hat, columns=['Beta Values'], index=['β'+ str(i) for i in range(1, k+1)]))

X_test = test_data[['GRE Score', 'GPA']].values
Y_hat = X_test.dot(beta_hat).round(3)
# print(Y_hat.flatten())

# Calculate SSE
residual_cap = Y_test_actual - Y_hat
sse = (residual_cap ** 2).sum()
print("SSE using GRE and GPA features is {0}\n".format(round(sse, 4)))