import os
from read_subject import get_personal_information
from read_sensor import get_respiban, read_empatica_csv, read_pkl, read_quest_csv

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