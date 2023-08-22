import os
from read_subject import get_personal_information
from read_sensor import read_pkl, read_quest_csv

""" 
Problems definition: 
1. To create a data processing pipeline for WESAD sensor data. 
2. The resulting datasets should be in consistent format. (Same column names and types and order)
3. The end output helps in easier analysing and using for building predictive mood model (classification).
4. The processed data should be clear and full with details to provide greater space for feature engineering / extraction.

Git repo: https://github.com/Zhejing-Chin/OnePlanet
Functions in separate files for easier management and code reusabiltiy. 
"""


path = "./WESAD"
subjects = next(os.walk(path))[1]

# ./WESAD/S2/S2_readme.txt
# personal_information = get_personal_information(path, subjects)
# print(personal_information)

# ./WESAD/S2/S2_respiban.txt = chest
# respiban = get_respiban(path, subjects[1:2])
# print(respiban)

# ./WESAD/S2/S2_E4_Data.zip = wrist
# read_empatica_zip(path, subjects[0])

# ./WESAD/S2/S2.pkl = synchronised chest + wrist
# print(len(read_pkl(path, subjects[1])['signal']['wrist']['TEMP']))
sync_c, sync_w = read_pkl(path, subjects[1])
print(sync_w)
# print()

# ./WESAD/S2/S2_quest.csv
# ground_truth = read_quest_csv(path, subjects[1])
# print()