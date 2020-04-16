import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import numpy as np
import sys
import collections
import itertools
from scipy import stats
from scipy.stats import mode
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

def check_stationarity(timeseries, plot=True):
    # Determing rolling statistics
    rolling_mean = timeseries.rolling(window=52, center=False).mean()
    rolling_std = timeseries.rolling(window=52, center=False).std()

    # Plot rolling statistics:
    if plot:
        original = plt.plot(timeseries.index.to_pydatetime(), timeseries.values, color='blue', label='Original')
        mean = plt.plot(rolling_mean.index.to_pydatetime(), rolling_mean.values, color='red', label='Rolling Mean')
        std = plt.plot(rolling_std.index.to_pydatetime(), rolling_std.values, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dickey_fuller_test = adfuller(timeseries, autolag='AIC')
    dfresults = pd.Series(dickey_fuller_test[0:4],
                          index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dickey_fuller_test[4].items():
        dfresults['Critical Value (%s)' % key] = value
    print(dfresults)
    return dfresults

def removeSeasonDecomposition(dataframe, title):
    dataframe.index=dataframe.index.to_timestamp()
    dataframe_log = np.log(dataframe)
    decomposition = seasonal_decompose(dataframe)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Select the most recent weeks
    dataframe_log_select = dataframe_log

    plt.subplot(411)
    plt.plot(dataframe_log_select.index.to_pydatetime(), dataframe_log_select.values, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(dataframe_log_select.index.to_pydatetime(), trend.values, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(dataframe_log_select.index.to_pydatetime(), seasonal.values, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(dataframe_log_select.index.to_pydatetime(), residual.values, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close()
    return plt

def removeSeasonDifferencing(dataframe):
    dataframe_log = np.log(dataframe)
    dataframe_log_diff = dataframe_log - dataframe_log.shift()
    plt.plot(dataframe_log_diff.index.to_pydatetime(), dataframe_log_diff.values)
    dataframe_log_diff.dropna(inplace=True)
    dfresults = check_stationarity(dataframe_log_diff)
    return dfresults

# if __name__ == "__main__":
    # check_stationarity()
    # removeSeasonDecomposition()
    # removeSeasonDifferencing()