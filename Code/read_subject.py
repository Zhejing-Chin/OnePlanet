import pandas as pd

def read_readme_txt(dataFolder, subject):
    path = f"{dataFolder}/{subject}/{subject}_readme.txt"
    with open(path) as f:        
        for line in f:
            line_ori = line
            line = line.lower()
            if "###" in line: continue
            elif "age" in line: 
                age = int(line.rsplit(": ", 1)[1].strip())
            elif "height" in line: 
                height = float(line.rsplit(": ", 1)[1].strip())
            elif "weight" in line:
                weight = float(line.rsplit(": ", 1)[1].strip())
            elif "gender" in line:
                gender = line.rsplit(": ", 1)[1].strip()
            elif "dominant" in line:
                hand = line.rsplit(": ", 1)[1].strip()
            elif "coffee today" in line:
                coffee_today = line.rsplit("? ", 1)[1].strip()
            elif "coffee within" in line:
                coffee_last_hr = line.rsplit("? ", 1)[1].strip()
            elif "sports today" in line:
                sports_today = line.rsplit("? ", 1)[1].strip()
            elif "smoker" in line:
                smoker = line.rsplit("? ", 1)[1].strip()
            elif "smoke within" in line:
                smoker_last_hr = line.rsplit("? ", 1)[1].strip()  
            elif "ill today" in line:
                ill_today = line.rsplit("? ", 1)[1].strip()    
            else:
                additional_notes = line_ori.strip()   
                
    row = {
        'id': subject,
        'age' : age, 
        'height_cm' : height, 
        'weight_kg' : weight,
        'gender' : gender, 
        'dominant_hand' : hand, 
        'coffee_today' : coffee_today,
        'coffee_last_hr' : coffee_last_hr, 
        'sports_today' : sports_today, 
        'smoker' : smoker,
        'smoke_last_hr' : smoker_last_hr, 
        'ill_today' : ill_today, 
        'additional_notes' : additional_notes}
        
    return row
    
    
def get_personal_information(dataFolder, subjects):
    personal_information = pd.DataFrame({'id':'', 
                                    'age':int(), 
                                    'height_cm':float(), 
                                    'weight_kg':float(), 
                                    'gender':'', 
                                    'dominant_hand':'',
                                    'coffee_today':'', 
                                    'coffee_last_hr':'', 
                                    'sports_today':'', 
                                    'smoker':'',
                                    'smoke_last_hr':'', 
                                    'ill_today':'', 
                                    'additional_notes':''}, index=[])
    
    for subject in subjects:
        row = read_readme_txt(dataFolder, subject)
        
        personal_information = personal_information.append(row,
            ignore_index = True)
        
    return personal_information