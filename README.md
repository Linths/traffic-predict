# Predict traffic

Predict the travel times for the Amsterdam environment using time series analysis.

## Abstract

Making predictions regarding the average travel time as accurately as possible for carriage travellers is highly desired. Various forecasting methods such as SARIMA and TBATS have been applied on adapted time series of traffic travel times. The data set used concerns a subsection of the roads in Amsterdam and its surroundings. For the experiments with the two models a small subsection of the available road sections was selected. Many strong seasonalities are found in the signal, such as daily commutes and the separation of work days and weekends. While the out-of-sample predictions are reasonably correct (SARIMA's mean absolute error is 1.58, TBATS 1.74), they are not ideal. While certain seasonalities (e.g. daily) seemed to be modeled correctly, others (e.g. weekly) give room for improvement. The research concludes with an in-depth discussion and suggestions for future work.

## Run it

Install the [Dutch national road traffic](https://www.ndw.nu/) data set. Preferably the data from June-July 2016 to ensure compatibility.
Make sure you have Python 3.x and common packages like `pandas`, `numpy`, `matplotlib.pyplot`, `statsmodels` and `sklearn`. Note you also need the package `tbats` which you can get [here](https://github.com/intive-DataScience/tbats).

Run the code:  
```> python main.py```

There is a lot of functionality (scattered in the code). Be aware you might need to (un)comment lines throughout the source to get exactly what you want.
