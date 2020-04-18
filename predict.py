import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tbats import TBATS, BATS
from sklearn.metrics import mean_absolute_error

def plotAcfPacf(ts_log_diff, nlags=20):
    lag_acf = acf(ts_log_diff, nlags=nlags)
    lag_pacf = pacf(ts_log_diff, nlags=nlags, method='ols')
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

def arima(ts, ts_log, ts_log_diff, p, d, q, forget_last, q_tuple=None, seasonal_order=None):
    last_steps = len(ts_log) #60 * 24
    new_steps = forget_last
    trainset = ts_log[:-forget_last]

    # RSS are the residual number of squares
    # AR
    model = ARIMA(trainset, order=(p, d, 0))
    results_AR = model.fit(disp=-1)  
    ts_log_diff.plot()
    results_AR.fittedvalues.plot()
    plt.title('AR RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff[:-forget_last])**2))
    plt.show()
    predicted = results_AR.forecast(steps=new_steps)
    # plt.plot(np.append(ts_log[-last_steps:], predicted[0]))
    # plt.plot(np.exp(np.append(ts_log[-last_steps:], predicted[0])), color='orange')
    plt.plot(range(0, last_steps), np.exp(ts_log).to_numpy(), color='blue')
    plt.plot(range(last_steps-forget_last, last_steps), np.exp(predicted[0]), color='orange')
    ci = predicted[2]
    ax = plt.gca()
    ax.fill_between(range(last_steps-forget_last, last_steps), np.exp(ci[:,0]), np.exp(ci[:,1]), color='b', alpha=.1)
    # xx = ts_log.iloc[-1].loc[:, "periodStart"].values #['periodStart']
    plt.ylim(-2, np.max(350))
    plt.axvline(x=last_steps-forget_last, color='red')
    plt.title(f"AR prediction of travel time (MAE: %.4f)"% mean_absolute_error(np.exp(ts_log).to_numpy()[-forget_last:], np.exp(predicted[0])))
    plt.show()
    plt.close()

    # MA
    model = ARIMA(trainset, order=(0, d, q))  
    results_MA = model.fit(disp=-1)  
    plt.plot(ts_log_diff.to_numpy())
    plt.plot(results_MA.fittedvalues.to_numpy(), color='red')
    plt.title('MA RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff[:-forget_last])**2))
    plt.show()
    predicted = results_MA.forecast(steps=new_steps)
    # plt.plot(np.append(ts_log[-last_steps:], predicted[0]))
    # plt.plot(np.exp(np.append(ts_log[-last_steps:], predicted[0])), color='orange')
    plt.plot(range(0, last_steps), np.exp(ts_log).to_numpy(), color='blue')
    plt.plot(range(last_steps-forget_last, last_steps), np.exp(predicted[0]), color='orange')
    ci = predicted[2]
    ax = plt.gca()
    ax.fill_between(range(last_steps-forget_last, last_steps), np.exp(ci[:,0]), np.exp(ci[:,1]), color='b', alpha=.1)
    plt.ylim(-2, np.max(350))
    plt.axvline(x=last_steps-forget_last, color='red')
    plt.title(f"MA prediction of travel time (MAE: %.4f)"% mean_absolute_error(np.exp(ts_log).to_numpy()[-forget_last:], np.exp(predicted[0])))
    plt.show()
    
    # ARIMA
    model = ARIMA(trainset, order=(p, d, q))  
    results_ARIMA = model.fit(disp=-1)  
    plt.plot(ts_log_diff.to_numpy())
    plt.plot(results_ARIMA.fittedvalues.to_numpy(), color='red')
    plt.title('ARIMA RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff[:-forget_last])**2))
    plt.show()
    predicted = results_ARIMA.forecast(steps=new_steps)
    # plt.plot(np.append(ts_log[-last_steps:-forget_last], predicted[0]))
    # plt.plot(np.exp(np.append(ts_log[-last_steps:-forget_last], predicted[0])), color='orange')
    plt.plot(range(0, last_steps), np.exp(ts_log).to_numpy(), color='blue')
    plt.plot(range(last_steps-forget_last, last_steps), np.exp(predicted[0]), color='orange')
    ci = predicted[2]
    ax = plt.gca()
    ax.fill_between(range(last_steps-forget_last, last_steps), np.exp(ci[:,0]), np.exp(ci[:,1]), color='b', alpha=.1)
    plt.ylim(-2, np.max(350))
    plt.axvline(x=last_steps-forget_last, color='red')
    plt.title(f"ARIMA prediction of travel time (MAE: %.4f)"% mean_absolute_error(np.exp(ts_log).to_numpy()[-forget_last:], np.exp(predicted[0])))
    plt.show()

    # # Show in-sample predictions
    # predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    # predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    # predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
    # predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    # predictions_ARIMA = np.exp(predictions_ARIMA_log)
    # plt.plot(ts.to_numpy(), color='blue')
    # plt.plot(predictions_ARIMA.to_numpy(), color='orange')
    # plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
    # plt.show()

    # Not currently using q_tuple in SARIMAX
    if seasonal_order == None:
        return

    # SARIMAX
    model = SARIMAX(trainset, trend='c', order=(p, d, q), seasonal_order=seasonal_order) # Doesn't seem to be a time trend
    results_SARIMAX = model.fit(disp=-1)  
    plt.plot(ts_log.to_numpy())
    plt.plot(results_SARIMAX.fittedvalues.to_numpy(), color='red')
    plt.title('SARIMA RSS: %.4f'% sum((results_SARIMAX.fittedvalues.to_numpy()-ts_log[:-forget_last].to_numpy())**2))
    plt.show()
    predicted = results_SARIMAX.forecast(steps=new_steps)
    # plt.plot(np.append(ts_log[-last_steps:-forget_last], predicted[0]))
    # plt.plot(np.exp(np.append(ts_log[-last_steps:-forget_last], predicted[0])), color='orange')
    plt.plot(range(0, last_steps), np.exp(ts_log).to_numpy(), color='blue')
    plt.plot(range(last_steps-forget_last, last_steps), np.exp(predicted), color='orange')
    # ci = predicted[2]
    # ax = plt.gca()
    # ax.fill_between(range(last_steps-forget_last, last_steps), np.exp(ci[:,0]), np.exp(ci[:,1]), color='b', alpha=.1)
    plt.ylim(-2, np.max(350))
    plt.axvline(x=last_steps-forget_last, color='red')
    plt.title(f"SARIMA prediction of travel time (MAE: %.4f)"% mean_absolute_error(np.exp(ts_log).to_numpy()[-forget_last:], np.exp(predicted)))
    plt.show()

def tbats(ts, ts_log, ts_log_diff, forget_last, periods):
    last_steps = len(ts_log) #60 * 24
    new_steps = forget_last
    trainset = ts_log[:-forget_last]

    # Fit the model
    estimator = TBATS(
        seasonal_periods=periods,
        use_arma_errors=False,  # shall try only models without ARMA
        use_box_cox=False  # will not use Box-Cox
    )
    model = estimator.fit(trainset)
    # In-sample
    plt.plot(ts_log.to_numpy())
    plt.plot(model.y_hat, color='red')
    plt.title('TBATS RSS: %.4f'% sum((model.y_hat-ts_log[:-forget_last].to_numpy())**2))
    plt.show()

    # Forecast ahead
    predicted = model.forecast(steps=forget_last, confidence_level=0.95)
    plt.plot(range(0, last_steps), np.exp(ts_log).to_numpy(), color='blue')
    plt.plot(range(last_steps-forget_last, last_steps), np.exp(predicted[0]), color='orange')
    ci = predicted[1]
    ax = plt.gca()
    ax.fill_between(range(last_steps-forget_last, last_steps), np.exp(ci['lower_bound']), np.exp(ci['upper_bound']), color='b', alpha=.1)
    plt.ylim(-2, np.max(350))
    plt.axvline(x=last_steps-forget_last, color='red')
    plt.title(f"TBATS prediction of travel time (MAE: %.4f)"% mean_absolute_error(np.exp(ts_log).to_numpy()[-forget_last:], np.exp(predicted[0])))
    plt.show()
    print(model.summary())
    # print(predicted)