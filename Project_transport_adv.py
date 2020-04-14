import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import numpy as np
from scipy import stats
from random import randint
import pickle
import predict

TT_FILE = "data/travel_times_compact.p"
TT_WATERGRAAFSMEER_FILE = "data/travel_times_compact_watergraafsmeer.p"

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
    speed_meta = pd.read_csv(r'../TRANSPORT/NDW/utwente snelheden groot amsterdam  1 dag met metadata_snelheid_00001.csv', chunksize=1000000, low_memory=False)

    # append each chunk df here
    intensity_meta_list = []
    travel_times_meta_list = []
    speed_meta_list = []

    # Each chunk is in df format
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

def checkStationarity(ts):
    # Check stationarity of time signal
    return

def predictTT(ts):
    # If you do not want to change intervals, put everything on False
    # If you want to change the signal to a certain interval, set only that one to True
    DAILY = False
    HOURLY = False
    SHORTLY = True # Every 5 minutes

    # plt.plot(ts, color='blue')
    ts = ts.replace(-1.0, np.nan) #0.000001)
    ts = ts.interpolate(method='time')
    # plt.plot(ts, color='orange')
    # plt.title("Interpolation")
    # plt.show()
    # plt.close()
    ts.index = pd.DatetimeIndex(ts.index).to_period('min') #, freq='m')
    if DAILY:
        ts = ts.resample('D').mean()
    elif HOURLY:
        ts = ts.resample('h').mean()
    elif SHORTLY:
        ts = ts.resample('5min').mean()
    ts.plot()
    plt.title("Travel time with interpolation (and possibly interval conversion)")
    plt.show()
    plt.close()
    # ts.dropna(inplace=True)
    # ts = ts.to_numpy()
    ts_log = np.log(ts)
    ts_log.index = ts.index
    ts_log.plot()
    plt.title("Travel time (log)")
    plt.show()
    plt.close()
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.index = ts.index
    ts_log_diff.dropna(inplace=True)
    # ts_log_diff = ts_log - np.roll(ts_log, 1)
    ts_log_diff.plot()
    plt.title("Travel time (log diff)")
    plt.show()
    plt.close()

    if DAILY:
        forget_last = 7
    elif HOURLY:
        forget_last = 24 * 7
    elif SHORTLY:
        forget_last = 12 * 23 * 7 #-276 -312 -10
    else:
        forget_last = 60 * 24 * 7
    predict.plotAcfPacf(ts_log_diff[:-forget_last])
    
    # predict.plotAcfPacf(ts_log_diff)
    if DAILY:
        p = 1
        q = 1
    elif HOURLY:
        p = 1 #9
        q = 1 #3
    elif SHORTLY:
        p = 2
        q = 2
    else:
        p = 3 #3 #2.6
        q = 3 #2 #2.35
    predict.arima(ts, ts_log, ts_log_diff, p, 1, q, forget_last)

def extractTrialLocation():
    # travel_times_comp = pd.read_pickle(r'data/travel_times_compact.p')
    # preprocessFlowData(travel_times_comp)
    travel_times_all = []
    for i in range(1, 5):
        travel_times_all.append(
            pd.read_csv(f'../TRANSPORT/NDW/utwente reistijden groot amsterdam  _reistijd_0000{i}.csv',
                        chunksize=1000000, low_memory=False))

    travel_concat_all = []
    for travel_times in travel_times_all:

        # Each chunk is in df format
        chunk_list = []
        for chunk in travel_times:
            # Once the data filtering is done, append the chunk to list
            chunk_list.append(chunk)
        # concat the list into dataframe
        travel_concat_all.append(pd.concat(chunk_list))

    print(travel_concat_all)

    travel_times_meta_list = []
    travel_times_meta = pd.read_csv(
        r'../TRANSPORT/NDW/utwente reistijden groot amsterdam  1 dag met metadata_reistijd_00001.csv',
        chunksize=1000000, low_memory=False)
    for chunk in travel_times_meta:
        # Once the data filtering is done, append the chunk to list
        travel_times_meta_list.append(chunk)
    # concat the list into dataframe
    travel_times_meta_concat = pd.concat(travel_times_meta_list)

    # travel_times_comp = pd.DataFrame(data = travel_times_comp)
    #
    # # merge two dataframes of metadata and regular data
    merged_travel_times = travel_times_meta_concat.merge(travel_concat_all, on='measurementSiteReference')
    print(merged_travel_times)

if __name__ == "__main__":
    print("Started")
    
    # Load all data
    # try:
    #     travel_time_data = pd.read_pickle(TT_FILE)
    #     print("Success!")
    # except:
    #     print("Couldn't find compact travel time file.")
    #     travel_time_data = storeTT()
    
    # Load watergraafsmeer
    try:
        travel_time_watergraafsmeer = pd.read_pickle(TT_WATERGRAAFSMEER_FILE)
        print("Success!")
    except:
        print("Couldn't find compact travel time file.")
        travel_time_watergraafsmeer = storeTTWatergraafsmeer()
    print(travel_time_watergraafsmeer.head)
    ts = travel_time_watergraafsmeer["avgTravelTime"]
    plt.plot(ts)
    plt.title("Original travel time of Watergraafsmeer")
    plt.show()
    plt.close()
    predictTT(ts)
    checkStationarity(ts)
