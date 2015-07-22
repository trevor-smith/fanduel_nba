"""
The goal of this file is to use the trained model from the previous Sunday
and to predict the nightly fantasy score of all the players playing
that night.  This file is a work in progress...only a shell is written
here so far.
"""

### IMPORTS ###
import pandas as pd
import numpy as np
import pickle

# load our pickled model

# load our data for the night
X = data[features]
y = data['fanduel_score']

# predict
model_final.predict(X, y)
