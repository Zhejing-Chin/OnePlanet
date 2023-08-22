import os
import pandas as pd
from read_subject import get_personal_information
from read_sensor import read_quest_csv, full_data_groundtruth

""" 
Problems definition: 
1. To create a data processing pipeline for WESAD sensor data. 
2. The resulting datasets should be in consistent format. (Same column names and types and order)
3. The end output helps in easier analysing and using for building predictive mood model (classification).
4. The processed data should be clear and full with details to provide greater space for feature engineering / extraction.

Assumptions:
1. The class labels are of study protocol (1-4).
    - higher completion of data
    - straightforward to interpret
    - self reports are subjective and might provide noise, higher complexity and influence to final output.

Git repo: https://github.com/Zhejing-Chin/OnePlanet
Functions in separate files for easier management and code reusabiltiy. 
"""


path = "./WESAD"
subjects = next(os.walk(path))[1]

# ./WESAD/S2/S2_readme.txt
personal_information = get_personal_information(path, subjects)
# print(personal_information.shape)

# ./WESAD/S2/S2_respiban.txt = chest
# respiban = get_respiban(path, subjects[1:2])
# print(respiban)

# ./WESAD/S2/S2.pkl = synchronised chest + wrist
# print(len(read_pkl(path, subjects[1])['signal']['wrist']['TEMP']))
# sensor = read_pkl(path, subjects[13])
# print(sensor)
# print()

# ./WESAD/S2/S2_quest.csv
# ground_truth = read_quest_csv(path, subjects[1])
# print()

# # Full synchronised sensor data with ground truth
# sensor_data = full_data_groundtruth(path, subjects)
# # print(sensor_data.shape)

# # The team able to analyse the full dataframe / with selected columns
# # combine personal information with sensor data
# full_data = personal_information.join(sensor_data, how='right', on="id")
# print(full_data.shape, full_data.columns)