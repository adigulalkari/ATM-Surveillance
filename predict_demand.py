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
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, shuffle= False, test_size = 0.2, random_state = 42)



xgb = XGBRegressor(objective = 'reg:squarederror' )
xgb.fit(Xtrain, ytrain)

# make predictions

xgb_pred = xgb.predict(Xtest)

xgbpred = pd.DataFrame(xgb_pred[-10:])

xgbpred.rename(columns = {0: 'xgb_predicted'}, inplace=True)

xgbpred = xgbpred.round(decimals=0)

xgbpred.index = d.index

xgbok = pd.concat([xgbpred, d], axis=1)

import pandas as pd

# convert index to DatetimeIndex
xgbok.index = pd.to_datetime(xgbok.index)

# compute accuracy
xgbok['accuracy'] = round(xgbok.apply(lambda row: row.xgb_predicted/row.total_amount_withdrawn *100, axis=1),2)

# format accuracy as percentage
xgbok['accuracy'] = pd.Series(["{0:.2f}%".format(val) for val in xgbok['accuracy']], index= xgbok.index)

# add day_of_week column
xgbok = xgbok.assign(day_of_week = lambda x: x.index.day_name())

print(xgbok)