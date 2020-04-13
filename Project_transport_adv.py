import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import numpy as np
from scipy import stats
from random import randint
import pickle

TRAVEL_TIME_COMPACT_FILE = "data/travel_times_compact.p"

def storeMinimalTravelTimeData():
    travel_time_data = []
    for i in range(1,5):
        print(f"==========File {i}==========")
        travel_time_data_current = pd.read_csv(f'../TRANSPORT/NDW/utwente reistijden groot amsterdam  _reistijd_0000{i}.csv', chunksize=1000000)
        chunk_list = []
        for chunk in travel_time_data_current:
            chunk = preprocessTravelTimeData(chunk)
            chunk_list.append(chunk)
        travel_time_data.append(pd.concat(chunk_list))
        # print(travel_time_data)
    travel_time_data = pd.concat(travel_time_data)
    pickle.dump(travel_time_data, open(TRAVEL_TIME_COMPACT_FILE, "wb"))
    return travel_time_data

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

    for travel_times in travel_concat_all:
        print(travel_times)

    # Each chunk is in df format
    for chunk in intensity_meta:
        # Once the data filtering is done, append the chunk to list
        intensity_meta_list.append(chunk)
    # concat the list into dataframe
    intensity_meta_concat = pd.concat(intensity_meta_list)

def preprocessTravelTimeData(data):
    data = data[["measurementSiteReference","index","periodStart","periodEnd","numberOfIncompleteInputs","minutesUsed","computationalMethod","travelTimeType","avgTravelTime","generatedSiteName","lengthAffected"]]
    data[["periodStart", "periodEnd"]] = data[["periodStart", "periodEnd"]].astype("datetime64")
    data[["numberOfIncompleteInputs","minutesUsed","avgTravelTime","lengthAffected"]] = data[["numberOfIncompleteInputs","minutesUsed","avgTravelTime","lengthAffected"]].astype("float32")
    data.index = pd.DatetimeIndex(data['periodStart'])
    return data

def preprocessFlowData(data):
    data = data[["measurementSiteReference","measurementSiteVersion","index","periodStart","numberOfIncompleteInputs","avgVehicleFlow","generatedSiteName"]]
    data[["measurementSiteReference", "numberOfIncompleteInputs", "avgVehicleFlow"]] = data[["measurementSiteReference", "numberOfIncompleteInputs", "avgVehicleFlow"]].astype('float32')
    data["periodStart"] = data["periodStart"].astype("datetime64")
    return data

if __name__ == "__main__":
    try:
        travel_time_data = pickle.load(open(TRAVEL_TIME_COMPACT_FILE, "rb"))
    except:
        travel_time_data = storeMinimalTravelTimeData()
    # travel_time_watergraafsmeer = travel_time_data.loc[travel_time_data.measurementSiteReference == "RWS01_MONIBAS_0011hrl0036ra0"]
    # plt.plot(travel_time_watergraafsmeer["avgTravelTime"])
    # plt.show()
