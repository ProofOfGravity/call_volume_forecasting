from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Hourly
import plotly.graph_objs as go
from plotly.offline import iplot
import pandas as pd
import numpy as np
from datetime import date
import holidays
from sklearn.model_selection import train_test_split

# Read in case log daily counts, including date
df = pd.read_csv('Case Log Counts Daily.csv')

# method to generate time lagged data for time lag predictions
# def generate_time_lags(df, n_lags):
#     df_n = df.copy()
#     for n in range(1, n_lags + 1):
#         df_n[f"lag{n}"] = df_n["Calls"].shift(n)
#     df_n = df_n.iloc[n_lags:]
#     return df_n
#
# input_dim = 100
#
# df_generated = generate_time_lags(df, input_dim)
# print(df_generated)
#
# df_generated.to_csv('test.csv')

# set the index to the 'Date' of the data
df = df.set_index(['Date'])
df.index = pd.to_datetime(df.index)

# create df_features, break 'Date' into components
df_features = (
    df
        .assign(day=df.index.day)
        .assign(month=df.index.month)
        .assign(day_of_week=df.index.dayofweek)
        .assign(week_of_year=df.index.week)
)


# turn input into cyclic features
def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}': lambda x: np.sin(2 * np.pi * (df[col_name] - start_num) / period),
        f'cos_{col_name}': lambda x: np.cos(2 * np.pi * (df[col_name] - start_num) / period)
    }
    return df.assign(**kwargs).drop(columns=[col_name])


df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 0)
df_features = generate_cyclical_features(df_features, 'month', 12, 1)
df_features = generate_cyclical_features(df_features, 'week_of_year', 52, 0)

# generate holiday features
us_holidays = holidays.US()


def is_holiday(date):
    date = date.replace(hour=0)
    return 1 if (date in us_holidays) else 0


def add_holiday_col(df, holidays):
    return df.assign(is_holiday=df.index.to_series().apply(is_holiday))


# add generated holidays to the dataset
df_features = add_holiday_col(df_features, us_holidays)

print(df_features)


# df_features.to_csv('features.csv')

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y


def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, 'Calls', 0.2)

print("X_train values: ", X_train)
print("X_val values: ", X_val)
print("X_test values: ", X_test)
print("y_train values: ", y_train)
print("y_val values: ", y_val)
print("y_test values: ", y_test)
