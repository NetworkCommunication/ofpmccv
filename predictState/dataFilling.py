import math
from datetime import datetime

import pandas as pd

# Data pre-processing: deleting data
def directDeleteData():
    input_file = '../../data/nteractive_formation_improve.csv'
    output_file = 'directDelData.csv'
    df = pd.read_csv(input_file)

    rows_to_keep = []
    for index, row in df.iterrows():
        if (index + 1) % 20 == 0:
            row = row.drop(["Local_X", "Local_Y", "v_Vel", "v_Acc"])
        rows_to_keep.append(row)

    new_df = pd.DataFrame(rows_to_keep)

    new_df.to_csv(output_file, index=False)

# fill data algorithm
def fillData(data, prev_index, next_index):
    deltaT = 0.1
    angle = math.atan((data.at[next_index, "Local_Y"] - data.at[prev_index, "Local_Y"]) / (
                data.at[next_index, "Local_X"] - data.at[prev_index, "Local_X"])) if data.at[prev_index, "Local_X"] != data.at[next_index, "Local_X"] else math.pi / 2
    if angle != 90:
        x = data.at[prev_index, "Local_X"] + data.at[prev_index, "v_Vel"] * math.cos(angle) * deltaT + 0.5 * (
                0.5 * (data.at[prev_index, "v_Acc"] + data.at[next_index, "v_Acc"]) * math.cos(
            angle)) * deltaT ** 2
        y = data.at[prev_index, "Local_Y"] + data.at[prev_index, "v_Vel"] * math.sin(angle) * deltaT + 0.5 * (
                    0.5 * (data.at[prev_index, "v_Acc"] + data.at[next_index, "v_Acc"]) * math.sin(angle)) * deltaT ** 2
    else:  # angle = 90,only y has movement
        x = data.at[prev_index, "Local_X"]
        y = data.at[prev_index, "Local_Y"] + data.at[prev_index, "v_Vel"] * deltaT + 0.5 * (
                    0.5 * (data.at[prev_index, "v_Acc"] + data.at[next_index, "v_Acc"]) * math.sin(angle)) * deltaT ** 2

    v = data.at[prev_index, "v_Vel"] + 0.5 * (data.at[prev_index, "v_Acc"] + data.at[next_index, "v_Acc"]) * deltaT

    a = (data.at[next_index, "v_Acc"] - data.at[prev_index, "v_Acc"]) / (2 * deltaT)
    return x, y, v, a


# Fill with 0
def fillDataBefP(data, prev_index, next_index):
    x = data.at[prev_index, "Local_X"]
    y = data.at[prev_index, "Local_Y"]
    v = data.at[prev_index, "v_Vel"]

    a = data.at[next_index, "v_Acc"]
    return x, y, v, a

def deleteDataFillZero():

    input_file = '../../data/interactive_formation_improve.csv'
    output_file = 'deleteDataFillZero.csv'

    df = pd.read_csv(input_file)

    rows_to_keep = []
    for index, row in df.iterrows():
        if (index + 1) % 20 == 0:
            row["Local_X"] = 0
            row["Local_Y"] = 0
            row["v_Vel"] = 0
            row["v_Acc"] = 0
        rows_to_keep.append(row)

    new_df = pd.DataFrame(rows_to_keep)

    new_df.to_csv(output_file, index=False)

# Algorithmic filling
def deleteDataFillAlg():
    input_file = 'deleteDataFillZero.csv'
    output_file = 'deleteDataFillAlg.csv'

    df = pd.read_csv(input_file)

    rows_to_keep = []
    for index, row in df.iterrows():
        if (index + 1) % 20 == 0:
            prev_index = index - 1
            next_index = index + 1
            x, y, v, a = fillData(df, prev_index, next_index)
            row["Local_X"] = x
            row["Local_Y"] = y
            row["v_Vel"] = v
            row["v_Acc"] = a
        rows_to_keep.append(row)

    new_df = pd.DataFrame(rows_to_keep)

    new_df.to_csv(output_file, index=False)

# Fill with data from the previous moment
def deleteDataFillBefP():
    input_file = 'deleteDataFillZero.csv'
    output_file = 'deleteDataFillBefP.csv'

    df = pd.read_csv(input_file)

    rows_to_keep = []
    for index, row in df.iterrows():
        if (index + 1) % 20 == 0:
            prev_index = index - 1
            next_index = index + 1
            x, y, v, a = fillDataBefP(df, prev_index, next_index)
            row["Local_X"] = x
            row["Local_Y"] = y
            row["v_Vel"] = v
            row["v_Acc"] = a
        rows_to_keep.append(row)

    new_df = pd.DataFrame(rows_to_keep)

    new_df.to_csv(output_file, index=False)

# Sort data
def sortData():

    data_file_path = 'data.csv'
    data = pd.read_csv(data_file_path)

    sorted_grouped_data = data.sort_values(by=[data.columns[1]]).groupby(data.columns[0], sort=False)

    sorted_data = pd.DataFrame()

    for group_name, group_data in sorted_grouped_data:
        sorted_data = pd.concat([sorted_data, group_data])

    output_file_path = 'sorted_data.csv'
    sorted_data.to_csv(output_file_path, index=False)

#  Change time format
def changeTimeFormat():

    data_file_path = 'test1.csv'
    data = pd.read_csv(data_file_path)

    def convert_to_timestamp(time_str):
        time_format = "%Y-%m-%d %H:%M:%S.%f"
        time_obj = datetime.strptime(time_str, time_format)
        timestamp_ms = int(time_obj.timestamp() * 1000)
        return timestamp_ms

    data[data.columns[1]] = data[data.columns[1]].apply(convert_to_timestamp)

    output_file_path = 'test1.csv'
    data.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    # sortData()
    # deleteData()
    # changeTimeFormat()
    # directDeleteData()
    # deleteDataFillZero()
    # deleteDataFillAlg()
    deleteDataFillBefP()