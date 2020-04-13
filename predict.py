import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

def plotAcfPacf(ts_log_diff):
    lag_acf = acf(ts_log_diff, nlags=40)
    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
    # Plot ACF
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    # Plot PACF
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()

def arima(ts, ts_log, ts_log_diff, p, d, q):
    last_steps = len(ts_log) #60 * 24
    new_steps = 60 * 48
    
    # RSS are the residual number of squares
    # AR
    model = ARIMA(ts_log, order=(p, d, 0))
    results_AR = model.fit(disp=-1)  
    plt.plot(ts_log_diff)
    plt.plot(results_AR.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
    plt.show()
    predicted = results_AR.forecast(steps=new_steps)
    plt.plot(np.append(ts_log[-last_steps:], predicted[0]))
    plt.plot(np.exp(np.append(ts_log[-last_steps:], predicted[0])), color='orange')
    plt.axvline(x=last_steps, color='red')
    plt.show()
    # MA
    model = ARIMA(ts_log, order=(0, d, q))  
    results_MA = model.fit(disp=-1)  
    plt.plot(ts_log_diff)
    plt.plot(results_MA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
    plt.show()
    predicted = results_MA.forecast(steps=new_steps)
    plt.plot(np.append(ts_log[-last_steps:], predicted[0]))
    plt.plot(np.exp(np.append(ts_log[-last_steps:], predicted[0])), color='orange')
    plt.axvline(x=last_steps, color='red')
    plt.show()
    # ARIMA
    model = ARIMA(ts_log, order=(p, d, q))  
    results_ARIMA = model.fit(disp=-1)  
    plt.plot(ts_log_diff)
    plt.plot(results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
    plt.show()
    predicted = results_ARIMA.forecast(steps=new_steps)
    plt.plot(np.append(ts_log[-last_steps:], predicted[0]))
    plt.plot(np.exp(np.append(ts_log[-last_steps:], predicted[0])), color='orange')
    plt.axvline(x=last_steps, color='red')
    plt.show()
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    plt.plot(ts)
    plt.plot(predictions_ARIMA)
    plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
    plt.show()