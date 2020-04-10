import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import numpy as np
from scipy import stats
from random import randint

def readFiles():
    # import data files
    # read the large csv file with specified chunksize
    intensity_meta = pd.read_csv(r'utwente intensiteiten groot amsterdam  1 dag met metadata (2)_intensiteit_00001.csv', chunksize=1000000, low_memory=False)

    travel_times_all = []
    for i in range(1,5):
        travel_times_all.append(pd.read_csv(f'utwente reistijden groot amsterdam  _reistijd_0000{i}.csv', chunksize=1000000, low_memory=False))

    travel_times_meta = pd.read_csv(r'utwente reistijden groot amsterdam  1 dag met metadata_reistijd_00001.csv', chunksize=1000000, low_memory=False)
    speed_meta = pd.read_csv(r'utwente snelheden groot amsterdam  1 dag met metadata_snelheid_00001.csv', chunksize=1000000, low_memory=False)

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

    # Each chunk is in df format
    for chunk in intensity_meta:
        # Once the data filtering is done, append the chunk to list
        intensity_meta_list.append(chunk)
    # concat the list into dataframe
    intensity_meta_concat = pd.concat(intensity_meta_list)

def preprocessFlowData(data):
    data = data[["measurementSiteReference","measurementSiteVersion","index","periodStart","numberOfIncompleteInputs","avgVehicleFlow","generatedSiteName"]]
    data[["measurementSiteReference", "numberOfIncompleteInputs", "avgVehicleFlow"]] = data[["measurementSiteReference", "numberOfIncompleteInputs", "avgVehicleFlow"]].astype('float32')
    data["periodStart"] = data["periodStart"].astype("datetime64")

if __name__ == "__main__":
    readFiles()