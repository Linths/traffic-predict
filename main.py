import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import numpy as np
from scipy import stats
from random import randint
import pickle
import predict
from seasonality import *

TT_FILE = "data/travel_times_compact.p"
TT_WATERGRAAFSMEER_FILE = "data/travel_times_compact_watergraafsmeer.p"
TT_META = "data/travel_times_meta.p"
TT_FINAL = "data/travel_times_final.p"

'''Takes the raw travel times data and stores it in a more efficient dataframe.'''
def storeTT():
    tt_all = []
    for i in range(1,5):
        print(f"==========File {i}==========")
        tt_current = pd.read_csv(f'../TRANSPORT/NDW/utwente reistijden groot amsterdam  _reistijd_0000{i}.csv', chunksize=1000000)
        chunk_list = []
        for chunk in tt_current:
            chunk = preprocessTT(chunk)
            chunk_list.append(chunk)

        tt_all.append(pd.concat(chunk_list))
        # print(tt_all)
    tt_all = pd.concat(tt_all)
    pickle.dump(tt_all, open(TT_FILE, "wb"))
    return tt_all

def storeTTWatergraafsmeer():
    tt_all = []
    for i in range(1,5):
        print(f"==========File {i}==========")
        tt = pd.read_csv(f'../TRANSPORT/NDW/utwente reistijden groot amsterdam  _reistijd_0000{i}.csv', chunksize=1000000)
        tt = pd.concat([preprocessTT(chunk) for chunk in tt])
        tt = tt.loc[(tt['measurementSiteReference'] == "RWS01_MONIBAS_0011hrl0036ra0") & (tt['index'] == "1001Z")]
        print(tt)
        tt_all.append(tt)
    tt_all = pd.concat(tt_all)
    # p = pickle.Pickler()
    # p.fast = True
    pickle.dump(tt_all, open(TT_WATERGRAAFSMEER_FILE, "wb"))
    return tt_all

def readFiles():
    # import data files
    # read the large csv file with specified chunksize
    intensity_meta = pd.read_csv(r'../TRANSPORT/NDW/utwente intensiteiten groot amsterdam  1 dag met metadata (2)_intensiteit_00001.csv', chunksize=1000000, low_memory=False)

    travel_times_all = []
    for i in range(1,5):
        travel_times_all.append(pd.read_csv(f'../TRANSPORT/NDW/utwente reistijden groot amsterdam  _reistijd_0000{i}.csv', chunksize=1000000, low_memory=False))

    travel_times_meta = pd.read_csv(r'../TRANSPORT/NDW/utwente reistijden groot amsterdam  1 dag met metadata_reistijd_00001.csv', chunksize=1000000, low_memory=False)
    #speed_meta = pd.read_csv(r'../TRANSPORT/NDW/utwente snelheden groot amsterdam  1 dag met metadata_snelheid_00001.csv', chunksize=1000000, low_memory=False)

    # append each chunk df here
    intensity_meta_list = []
    travel_times_meta_list = []
    speed_meta_list = []

    # # Each chunk is in df format
    for chunk in intensity_meta:
        # Once the data filtering is done, append the chunk to list
        intensity_meta_list.append(chunk)
    # concat the list into dataframe
    intensity_meta_concat = pd.concat(intensity_meta_list)

    travel_concat_all = []
    for travel_times in travel_times_all:

        # Each chunk is in df format
        chunk_list = []
        for chunk in travel_times:
            # Once the data filtering is done, append the chunk to list
            chunk_list.append(chunk)
        # concat the list into dataframe
        travel_concat_all.append(pd.concat(chunk_list))

    # for travel_times in travel_concat_all:
    #     print(travel_times)

    # Each chunk is in df format
    for chunk in travel_times_meta:
        # Once the data filtering is done, append the chunk to list
        travel_times_meta_list.append(chunk)
    # concat the list into dataframe
    travel_times_meta_concat = pd.concat(travel_times_meta_list)
    pickle.dump(travel_times_meta_concat, open(TT_META, "wb"))

def preprocessTT(data):
    data = data.loc[:, ["measurementSiteReference","index","periodStart","periodEnd","numberOfIncompleteInputs","minutesUsed","computationalMethod","travelTimeType","avgTravelTime","generatedSiteName","lengthAffected"]]
    data.loc[:, ["periodStart", "periodEnd"]] = data[["periodStart", "periodEnd"]].astype("datetime64")
    data.loc[:, ["numberOfIncompleteInputs","minutesUsed","avgTravelTime","lengthAffected"]] = data[["numberOfIncompleteInputs","minutesUsed","avgTravelTime","lengthAffected"]].astype("float32")
    data.index = pd.DatetimeIndex(data['periodStart'])
    return data

def preprocessTTLighter(data):
    data = data.loc[:, ["measurementSiteReference","index","periodStart","avgTravelTime","generatedSiteName","lengthAffected"]]
    data.loc[:, ["periodStart"]] = data[["periodStart"]].astype("datetime64")
    data.loc[:, ["avgTravelTime","lengthAffected"]] = data[["avgTravelTime","lengthAffected"]].astype("float32")
    data.index = pd.DatetimeIndex(data['periodStart'])
    return data

def preprocessFlowData(data):
    data = data[["measurementSiteReference","measurementSiteVersion","index","periodStart","numberOfIncompleteInputs","avgVehicleFlow","generatedSiteName"]]
    data[["measurementSiteReference", "numberOfIncompleteInputs", "avgVehicleFlow"]] = data[["measurementSiteReference", "numberOfIncompleteInputs", "avgVehicleFlow"]].astype('float32')
    data["periodStart"] = data["periodStart"].astype("datetime64")
    return data

def predictTT(ts):
    # If you do not want to change intervals, put everything on False
    # If you want to change the signal to a certain interval, set only that one to True
    DAILY = False
    HOURLY = True
    HALFHOURLY = False
    SHORTLY = False # Every 5 minutes

    plt.plot(ts.index.to_pydatetime(), ts, color='blue')
    ts = ts.replace(-1.0, np.nan) #0.000001)
    ts = ts.interpolate(method='time')
    plt.plot(ts.index.to_pydatetime(), ts, color='orange')
    plt.title("Interpolated travel time of Watergraafsmeer")
    plt.show()
    plt.close()
    ts.index = pd.DatetimeIndex(ts.index).to_period('min') #, freq='m')

    fs = 1/60
    if DAILY:
        forget_last = 7
    elif HOURLY:
        forget_last = 24 * 7
    elif HALFHOURLY:
        forget_last = 24 * 7 * 2
    elif SHORTLY:
        forget_last = 12 * 23 * 7 #-276 -312 -10
    else:
        forget_last = 60 * 24 * 7
    printPopularFrequencies(ts[:-forget_last], fs, "DFT of travel time of Watergraafsmeer (train data)")

    if DAILY:
        ts = ts.resample('D').mean()
        fs = 1/(60*60*24)
    elif HOURLY:
        ts = ts.resample('h').mean()
        fs = 1/(60*60)
    elif HALFHOURLY:
        ts = ts.resample('30min').mean()
        fs = 1/(60*30)
    elif SHORTLY:
        ts = ts.resample('5min').mean()
        fs = 1/(60*5)

    check_stationarity(ts, plot=False)
      
    ts.plot()
    plt.title("Travel time of Watergraafsmeer with interpolation\nand interval conversion")
    plt.show()
    plt.close()

    printPopularFrequencies(ts[:-forget_last], fs, "DFT of hourly travel time of Watergraafsmeer (train data)")
    
    ts_log = np.log(ts)
    ts_log.index = ts.index
    ts_log.plot()
    plt.title("Hourly travel time of Watergraafsmeer (log)")
    plt.show()
    plt.close()

    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.index = ts.index
    ts_log_diff.dropna(inplace=True)
    ts_log_diff.plot()
    plt.title("Hourly travel time of Watergraafsmeer (log diff)")
    plt.show()
    plt.close()

    check_stationarity(ts_log_diff, plot=False)
    removeSeasonDecomposition(ts[forget_last:], "Decomposition of the hourly travel time of Watergraafsmeer (train data)")
    # removeSeasonDecomposition(ts[:forget_last], "Decomposition of the hourly travel time of Watergraafsmeer\\of one week")

    predict.plotAcfPacf(ts_log_diff[:-forget_last])
    
    # predict.plotAcfPacf(ts_log_diff)
    if DAILY:
        p = 1
        q = 1
        q_tuple = None
    elif HOURLY:
        p = 1 #0 #19 #1 #9
        q = 1 #3
        q_tuple = 8*[0]
        q_tuple[1-1] = 1
        q_tuple[8-1] = 1
        q_tuple = tuple(q_tuple)
        periods = (8, 24, 4.8, 24*7)
        seasonal_order = (1,1,1,24) #24*7
    elif HALFHOURLY:
        p = 1
        q = 1
        q_tuple = None
        seasonal_order = None
    elif SHORTLY:
        p = 2
        q = 2
        q_tuple = None
        seasonal_order = None
    else:
        p = 3 #3 #2.6
        q = 3 #2 #2.35
        # Assume daily and weekly patterns
        # Periodicity is 60*24 and 60*24*7
        # q_tuple = 60*24*7*[0]
        # q_tuple = 60*24*[0]
        q_tuple = 60*[0]
        q_tuple[3-1] = 1
        q_tuple[60-1] = 1
        # q_tuple[60*24-1] = 1
        # q_tuple[60*24*7-1] = 1
        q_tuple = tuple(q_tuple)
        periods = (8*60, 24*60, 4.8*60, 24*7*60)
        seasonal_order = None
    predict.arima(ts, ts_log, ts_log_diff, p, 1, q, forget_last, seasonal_order=seasonal_order) #q_tuple)
    predict.tbats(ts, ts_log, ts_log_diff, forget_last, periods)

def applyFFT(ts, fs=1/60):
    signal = ts.copy()
    signal = signal - np.mean(signal)
    f = np.fft.fft(signal)
    x = np.fft.fftfreq(len(signal), d=1/fs)
    return x, abs(f)/len(signal)

def printPopularFrequencies(ts, fs, title):
    x, y = applyFFT(ts, fs)
    i = 0
    show = 10
    for popular_freq, amplitude in sorted(zip(x,y), reverse=True, key=lambda x: x[1])[:2*show]:
        if i%2 == 0:
            print(f"#{i//2+1} most popular frequency is every {(1/popular_freq)/60/60} hours [{amplitude}]")
        i += 1
    # Only plot one side of the mirrored graph
    mid = len(x)//2
    x = x[:mid]
    y = y[:mid]
    plt.plot(x, y)
    ax = plt.gca()
    ax.set_xlim(0)
    plt.title(title)
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("Started")

    travel_time_watergraafsmeer = pickle.load(open("data/travel_times_compact_watergraafsmeer.p", "rb"))
    ts = travel_time_watergraafsmeer["avgTravelTime"]
    # check_stationarity(ts)
    # removeSeasonDifferencing(ts)
    
    predictTT(ts)
 

    # plt.plot(ts)
    # plt.title("Original travel time of Watergraafsmeer")
    # plt.show()
    # plt.close()
    #
    #


    # final_travel_times = pickle.load(open("data/travel_times_final.p", "rb"))
    # print(final_travel_times.columns)

    # plt.plot(final_travel_times["avgTravelTime"])
    # plt.title("Original travel time of Watergraafsmeer")
    # plt.show()
    # plt.close()
    # #print(final_travel_times["avgTravelTime"])
    # predictTT(final_travel_times["avgTravelTime"])
    # checkStationarity(final_travel_times["avgTravelTime"])

    # Load all data
    # try:
    #     travel_time_data = pd.read_pickle(TT_FILE)
    #     print("Success!")
    # except:
    #     print("Couldn't find compact travel time file.")
    #     travel_time_data = storeTT()
    
    # # Load watergraafsmeer
    # try:
    #     travel_time_watergraafsmeer = pd.read_pickle(TT_WATERGRAAFSMEER_FILE)
    #     print("Success!")
    # except:
    #     print("Couldn't find compact travel time file.")
    #     travel_time_watergraafsmeer = storeTTWatergraafsmeer()
    # print(travel_time_watergraafsmeer.head)

    # try:
    #     travel_time_meta = pd.read_pickle(TT_META)
    #     print("Success!")
    # except:
    #     print("Couldn't find compact travel time file.")
    #     travel_time_meta = readFiles()
    #readFiles()

