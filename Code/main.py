import os
from read_subject import get_personal_information

path = "./WESAD"
subjects = next(os.walk(path))[1]

# ./WESAD/S2/S2_readme.txt
personal_information = get_personal_information(path, subjects)
print(personal_information)

