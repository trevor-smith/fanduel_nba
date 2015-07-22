"""
This file transforms the historical data from 2012-2014.  It also transforms
the files that I will be receiving daily.  The input files are expected to be
CSV, but I will also have to load historical data too.  For now, this
historical data will come from the CSV file, but I would like to transfer
this over to Postres
"""

### IMPORTS ###
import pandas as pd
import numpy as np

### HISTORICAL TRANSFORM ###
df = pd.read_csv('2012-2014-Table 1.csv')
pd.set_option('display.max_columns', 100)

# create fanduel score
    """
    this function calculates the fanduel score
    points = points
    rebounds = 1.2x
    assists = 1.5x
    blocks = 2.0x
    steals = 2.0x
    turnovers = -1.0x
    """
df['fanduel_score'] = df.apply(lambda x: x['PTS'] + x['TOT']*1.2 + x['A']*1.5 +
                               x['BL']*2 + x['ST']*2 - x['TO'], axis=1)

# need to sort data first...come up with robust way to do this

# create moving averages
def add_moving_averages(column, period):
    """
    this function creates a moving average for a column for a specified
    period of time

    column: the column you want to create a moving average for ('' to enclose)
    period: the number of previous games to be used for the moving average
    """
    column = str(column)
    new_column_name = "moving_average_"+column
    period = int(period)
    df[new_column_name] = pd.rolling_mean(df[column], period)

