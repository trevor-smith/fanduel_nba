"""
The goal of this file is to document how I predict fanduel scores using only
 basic data from nbastuffer.com.  The data set has only been augmented
 slightly by changing column names to be more readable and I've also added in
  player positions which I found on nba.com
"""

### IMPORTS ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split

### DATA PREP ###
df = pd.read_csv('2012-2014-Table 1.csv')
pd.set_option('display.max_columns', 100)
%matplotlib inline

# adding fanduel score
df['fanduel_score'] = df.apply(lambda x: x['PTS'] + x['TOT']*1.2 + x['A']*1.5 +
                               x['BL']*2 + x['ST']*2 -x['TO'], axis=1)

# creating dummy columns
home_dummies = pd.get_dummies(df.VENUE, prefix='venue')
position_dummies = pd.get_dummies(df.Position, prefix='position')
team_dummies = pd.get_dummies(df.OPP_TEAM, prefix='Opponent')

# combining them all together to form new data set
data = pd.concat([df, home_dummies, position_dummies, team_dummies], axis=1)

# our features
features = ['MIN', 'position_C', u'position_PF', u'position_PG', u'position_SF'
, u'position_SG', u'position_Unknown', 'TO', 'PF', 'venue_H', 'venue_R'
,'Opponent_Bos', u'Opponent_Bro', u'Opponent_Cha', u'Opponent_Chi'
, u'Opponent_Cle', u'Opponent_Dal', u'Opponent_Den', u'Opponent_Det'
, u'Opponent_Gol', u'Opponent_Hou', u'Opponent_Ind', u'Opponent_Lac'
, u'Opponent_Lal', u'Opponent_Mem', u'Opponent_Mia', u'Opponent_Mil'
, u'Opponent_Min', u'Opponent_Nor', u'Opponent_Nyk', u'Opponent_Okc'
, u'Opponent_Orl', u'Opponent_Phi', u'Opponent_Pho', u'Opponent_Por'
, u'Opponent_Sac', u'Opponent_San', u'Opponent_Tor', u'Opponent_Uta'
, u'Opponent_Was']

# split data into train and test
train, test = train_test_split(data3, train_size = 0.8)

x_train = train[features]
x_test = test[features]

y_train = train['fanduel_score']
y_test = test['fanduel_score']

### PREDICTIONS ###

# linear regression
lm = LinearRegression()
lm.fit(x_train, y_train)
print lm.score(x_test, y_test)
print zip(features, lm.coef_)
##### score = 0.691

# random forest
rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1, max_features='sqrt')
rf.fit(x_train, y_train)
print rf.score(x_test, y_test)
print zip(features, rf.coef_)
##### score = 0.667

# svm - linear kernel
svr_linear = SVR(kernel='linear', C=.5)
svr_linear.fit(x_train, y_train)
print svr_linear.score(x_train, y_train)
print zip(features, svr_linear.coef_)
##### score = 0.686

# svm - rbf kernel
svr_rbf = SVR(kernel='rbf', C=.5)
svr_rbf.fit(x_train, y_train)
print svr_rbf.score(x_train, y_train)
print zip(features, svr_rbf.coef_)
##### score = 0.700

# let's transform our dependent variable
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)
# fill in nan and inf
y_test_log = np.nan_to_num(y_test_log)
y_train_log = np.nan_to_num(y_train_log)

# svm - rbf kernel - log transform
svr_rbf = SVR(kernel='rbf', C=.5)
svr_rbf.fit(x_train, y_train_log)
print svr_rbf.score(x_train, y_train_log)
print zip(features, svr_rbf.coef_)
##### score = 0.700

### RESIDUALS ###
y_predicted = model_SVR.predict(x_test)
residuals = y_test - y_predicted
residuals.hist(bins=20)
# plot looks normally distributed...good.

### MOVING AVERAGES ###
data3['past_1'] = pd.rolling_mean(data3['fanduel_score'], 1)
data3['past_3'] = pd.rolling_mean(data3['fanduel_score'], 3)
data3['past_5'] = pd.rolling_mean(data3['fanduel_score'], 5)
data3['past_10'] = pd.rolling_mean(data3['fanduel_score'], 10)

train, test = train_test_split(data4, train_size = 0.8)

features6 = ['past_1', 'past_3', 'past_5', 'past_10','MIN', 'position_C',
 u'position_PF', u'position_PG', u'position_SF', u'position_SG',
 u'position_Unknown', 'TO', 'PF', 'venue_H', 'venue_R','Opponent_Bos',
  u'Opponent_Bro', u'Opponent_Cha', u'Opponent_Chi',u'Opponent_Cle',
  u'Opponent_Dal', u'Opponent_Den', u'Opponent_Det', u'Opponent_Gol',
  u'Opponent_Hou', u'Opponent_Ind',u'Opponent_Lac', u'Opponent_Lal',
  u'Opponent_Mem', u'Opponent_Mia', u'Opponent_Mil', u'Opponent_Min',
  u'Opponent_Nor', u'Opponent_Nyk', u'Opponent_Okc', u'Opponent_Orl',
   u'Opponent_Phi', u'Opponent_Pho',u'Opponent_Por', u'Opponent_Sac',
   u'Opponent_San', u'Opponent_Tor', u'Opponent_Uta', u'Opponent_Was']

x_train = train[features6]
x_test = test[features6]

y_train = train['fanduel_score']
y_test = test['fanduel_score']

model_SVR_rbf = SVR(kernel='rbf', C=.5)
model_SVR_rbf.fit(x_train, y_train)
model_SVR_rbf.score(x_test, y_test)
### Score: 0.938

