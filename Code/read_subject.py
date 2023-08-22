import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

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
        row = pd.DataFrame([read_readme_txt(dataFolder, subject)])
        
        personal_information = pd.concat([personal_information, row], ignore_index = True)
    
    personal_information = personal_information.set_index('id')    
    
    # EDA
    percent_missing = personal_information.isnull().sum() * 100 / len(personal_information)
    print(percent_missing)
     
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 6))
    sns.countplot(data=personal_information, x="gender", ax=axes[0,0])
    personal_information.groupby('gender').age.plot(kind='kde', legend=True, ax=axes[0,1])
    personal_information.groupby('gender').height_cm.plot(kind='kde', legend=True, ax=axes[0,2])
    personal_information.groupby('gender').weight_kg.plot(kind='kde', legend=True, ax=axes[0,3])
    sns.countplot(data=personal_information, x="gender", hue="dominant_hand", ax=axes[1,0])
    sns.countplot(data=personal_information, x="gender", hue="coffee_today", ax=axes[1,1])
    sns.countplot(data=personal_information, x="gender", hue="coffee_last_hr", ax=axes[1,2])
    sns.countplot(data=personal_information, x="gender", hue="sports_today", ax=axes[1,3])
    sns.countplot(data=personal_information, x="gender", hue="smoker", ax=axes[2,0])
    sns.countplot(data=personal_information, x="gender", hue="smoke_last_hr", ax=axes[2,1])
    sns.countplot(data=personal_information, x="gender", hue="ill_today", ax=axes[2,2])
    fig.delaxes(axes[2][3])
    plt.tight_layout()
    plt.savefig('./EDA/personal_information.png')
    # plt.show()
    
    personal_information.to_csv('./Data/personal_information.csv')

    
    return personal_information

# Preprocess the "additional notes" column
def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove stopwords and apply lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Join the lemmatized tokens back into a sentence
    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text