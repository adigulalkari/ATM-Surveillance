import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

data = pd.read_csv('atm_data_m2.csv')

# Copy the original data
features = data.copy()

# Select the numeric columns
numeric_subset = data[['trans_date_set','trans_month','trans_year','prevweek_mean', 'total_amount_withdrawn']]

# Select the categorical columns
# dropped atm_name
categorical_subset = data[['weekday','festival_religion', 'working_day',  'holiday_sequence']]

# One hot encoding
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

X = features.copy().drop(columns = ['total_amount_withdrawn', 'trans_date_set', 'trans_month','trans_year', 'working_day_H', 'working_day_W'])
y = features['total_amount_withdrawn'].copy()

# Split the data into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, shuffle= False, test_size = 0.2, random_state = 42)

# Extract the last 10 values of total_amount_withdrawn
d = pd.DataFrame(data['total_amount_withdrawn']).tail(10)

# Initialize the XGBRegressor model with mean squared error as objective function
xgb = XGBRegressor(objective = 'reg:squarederror')

# Fit the model on the training data
xgb.fit(Xtrain, ytrain)

# make predictions on the test data
xgb_pred = xgb.predict(Xtest)

# Get the last 10 values of the predicted amounts
xgbpred = pd.DataFrame(xgb_pred[-10:])

# Rename the column to 'xgb_predicted'
xgbpred.rename(columns = {0: 'xgb_predicted'}, inplace=True)

# Round the predicted values to the nearest integer
xgbpred = xgbpred.round(decimals=0)

# Set the index of the predicted values to be the same as the last 10 values of total_amount_withdrawn
xgbpred.index = d.index

# Concatenate the predicted values and the last 10 values of total_amount_withdrawn
xgbok = pd.concat([xgbpred, d], axis=1)

# Convert the index to a DatetimeIndex
xgbok.index = pd.to_datetime(xgbok.index)

# Compute the accuracy of the predictions
xgbok['accuracy'] = round(xgbok.apply(lambda row: row.xgb_predicted/row.total_amount_withdrawn *100, axis=1), 2)

# Format the accuracy as percentage
xgbok['accuracy'] = pd.Series(["{0:.2f}%".format(val) for val in xgbok['accuracy']], index= xgbok.index)

# Add a 'day_of_week' column
xgbok = xgbok.assign(day_of_week = lambda x: x.index.day_name())

# Print the dataframe with the predicted values, actual values, accuracy, and day of the week
print(xgbok)

# Compute the mean of the actual values and predicted values
mean_xgb_test = ytest.mean()
mean_xgb_pred = xgb_pred.mean()

# Print the mean of actual values and predicted values
print(mean_xgb_test, mean_xgb_pred)

# Compute the accuracy of the model
accuracy = (mean_xgb_test / mean_xgb_pred)*100
print('accuracy:', round(accuracy, 2) )


# prediction_date = pd.Timestamp('2023-04-01')
# prediction_features  = pd.DataFrame({'day_of_week': [prediction_date.day_name()],
#                                     'atm_name': 'Mount Road ATM',
#                                     'festival_religion': 'NH',
#                                     'working_day': 'W',
#                                     'holiday_sequence': 'WWW',
#                                     'trans_date_set': 1,
#                                     'trans_month': 1,
#                                     'trans_year':2023,
#                                     'preweek_mean':688670.0,
#                                     'total_amount_withdrawn':754400})

# prediction = xgb.predict(prediction_features)

# print("Predicted value for {}: {}".format(prediction_date, prediction[0]))