"""
The goal of this file is to train the optimal model for predicting fanduel
scores.  This file should be run once a week on a Sunday.  All games the
following week will use this model to predict fanduel scores
"""

### IMPORTS ###
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# data will come from the 'daily_data_transform.py' file
# the dataframe will be called 'data'

features = ['past_1', 'past_3', 'past_5', 'past_10', 'MIN', 'position_C',
            u'position_PF', u'position_PG', u'position_SF', u'position_SG',
            u'position_Unknown', 'TO', 'PF', 'venue_H', 'venue_R',
            u'Opponent_Bos', u'Opponent_Bro', u'Opponent_Cha', u'Opponent_Chi',
            u'Opponent_Cle',u'Opponent_Dal', u'Opponent_Den', u'Opponent_Det',
            u'Opponent_Gol', u'Opponent_Hou', u'Opponent_Ind', u'Opponent_Lac',
            u'Opponent_Lal', u'Opponent_Mem', u'Opponent_Mia', u'Opponent_Mil',
            u'Opponent_Min', u'Opponent_Nor', u'Opponent_Nyk', u'Opponent_Okc',
            u'Opponent_Orl', u'Opponent_Phi', u'Opponent_Pho', u'Opponent_Por',
            u'Opponent_Sac', u'Opponent_San', u'Opponent_Tor', u'Opponent_Uta',
            u'Opponent_Was']

# using crossval to validate our model
X = data[features]
y = data['fanduel_score']

from sklearn.cross_validation import cross_val_score
model = SVR(kernel='rbf', gamma=0, C=.6, epsilon=0)
scores = cross_validation.cross_val_score(model, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# now training our full model
model_final = SVR(kernel='rbf', gamma=0, C=.6, epsilon=0)
model_final.fit(X,y)
