import pandas as pd
import math
from zipfile import ZipFile
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks



def read_respiban_txt(dataFolder, subject):
    vcc=3
    chan_bit=2**16
    Cmin = 28000 
    Cmax = 38000
    
    path = f"{dataFolder}/{subject}/{subject}_respiban.txt"
    rows = []
    with open(path) as f:    
        next(f)
        next(f)
        # header = json.loads(f.readline().strip()[2:])
        next(f)    
        for line in f:
            values = line.strip().split()
            
            ecg = ((float(values[2])/chan_bit-0.5)*vcc)
            eda = (((float(values[3])/chan_bit)*vcc)/0.12)
            emg = ((float(values[4])/chan_bit-0.5)*vcc)
            vout = (float(values[5])*vcc)/(chan_bit-1.) 
            rntc = ((10**4)*vout)/(vcc-vout) 
            # assume log base e
            temp = -1
            try:
                temp = - 273.15 + 1./(1.12764514*(10**(-3)) + 2.34282709*(10**(-4))*math.log(rntc) + 8.77303013*(10**(-8))*(math.log(rntc)**3))
            except ValueError:
                print(values[0], rntc)
            xyz = [(float(x)-Cmin)/(Cmax-Cmin)*2-1 for x in values[6:9]]
            resp = (float(values[9]) / chan_bit - 0.5) * 100
            
            rows.append({'id':subject, 
            'seq':values[0],
            'ecg':ecg, 
            'eda':eda, 
            'emg':emg, 
            'temp':temp,
            'xyz_1':xyz[0],       
            'xyz_2':xyz[1], 
            'xyz_3':xyz[2], 
            'respiration':resp})
        
    return rows
        
        
def get_respiban(dataFolder, subjects):
    respiban = pd.DataFrame({'id':'', 
                             'seq':int(),
                             'ecg':float(), 
                             'eda':float(), 
                             'emg':float(), 
                             'temp':float(),
                             'xyz_1':float(),       
                             'xyz_2':float(), 
                             'xyz_3':float(), 
                             'respiration':float()}, index=[])
    
    for subject in subjects:
        rows = read_respiban_txt(dataFolder, subject)
        
        respiban = respiban.append(rows,
            ignore_index = True)
        
    return respiban

def unzip_empatica(dataFolder, subject):
    path = f"{dataFolder}/{subject}"
    # print(path)
    check_unzipped = next(os.walk(path))[1]
    if not f"{subject}_E4_Data" in check_unzipped:
        zipfile = f"{dataFolder}/{subject}/{subject}_E4_Data.zip"
    # print(path)
        with ZipFile(zipfile, 'r') as zObject:
            zObject.extractall(path+f"/{subject}_E4_Data")
            
def read_empatica_csv(dataFolder, subject):
    # make sure folder is unzipped
    unzip_empatica(dataFolder, subject)
    
    csv_files = ['ACC', 'BVP', 'EDA', 'TEMP']
    
def read_pkl(dataFolder, subject):
    path = f"{dataFolder}/{subject}/{subject}.pkl"
    sync = pd.read_pickle(path) 
    # ACC, ECG, EDA, EMG, Resp, Temp 
    acc = [list(x) for x in zip(*sync['signal']['chest']['ACC'])]
    length = len(acc[0])
    # print(length)
    # print(sync['signal']['chest'].keys())
    chest = {'xyz_1': acc[0],
             'xyz_2': acc[1],
             'xyz_3': acc[2],
             'ecg': sync['signal']['chest']['ECG'].reshape(length),
             'emg': sync['signal']['chest']['EMG'].reshape(length),
             'eda': sync['signal']['chest']['EDA'].reshape(length),
             'resp': sync['signal']['chest']['Resp'].reshape(length),
             'temp': sync['signal']['chest']['Temp'].reshape(length),
             'label': sync['label']}
    # print(chest)
    respiban = pd.DataFrame.from_dict(chest)
    respiban.insert(loc=0, column='id', value=subject)
    
    acc_chest = acc

    
    # ACC, BVP, EDA, TEMP
    acc = [list(map(float, x)) for x in zip(*sync['signal']['wrist']['ACC'])]
    # acc[0] = 
    #64hz
    bvp = sync['signal']['wrist']['BVP'].reshape(sync['signal']['wrist']['BVP'].size)
    #4hz
    eda = sync['signal']['wrist']['EDA'].reshape(sync['signal']['wrist']['EDA'].size)
    temp = sync['signal']['wrist']['TEMP'].reshape(sync['signal']['wrist']['TEMP'].size)
    # print(eda.size, bvp.size, temp.size)
    
    acc = [np.repeat(i, repeats=2, axis=0) * (1/64) for i in acc]
    eda = np.repeat(eda, repeats=16, axis=0)
    temp = np.repeat(temp, repeats=16, axis=0)
    
    print(acc[0].size, eda.size, bvp.size, temp.size)
    wrist = {'xyz_1': acc[0],
             'xyz_2': acc[1],
             'xyz_3': acc[2],
             'bvp': bvp,
             'eda': eda,
             'temp': temp}
    empatica = pd.DataFrame.from_dict(wrist)
    empatica.insert(loc=0, column='id', value=subject)
    
    acc_wrist = acc
    # plot_acceleration(acc_chest, acc_wrist)
    
    cross_corr_result = calculate_cross_correlation(acc_chest, acc_wrist)
    max_correlation_index = np.argmax(cross_corr_result)
    time_offset = max_correlation_index - (len(acc_chest) - 1)
    
    # assume label 4 link to both med 1 and med 2 in ground truth
    
    return respiban, empatica

def calculate_cross_correlation(signal_x, signal_y):
    cross_corr = np.correlate(signal_x, signal_y, mode='full')
    return cross_corr



def plot_acceleration(acc_chest, acc_wrist):
    acceleration = [acc_wrist, acc_chest]
    labels = ['X', 'Y', 'Z']

    plt.figure(figsize=(12, 6))

    for i in range(2): 
        plt.subplot(1, 2, i+1)
        for j in range(3):
            plt.plot(acceleration[i][j], label=labels[j])
        plt.title(('Wrist' if i==0 else 'Chest') + ' Acceleration Data')
        plt.xlabel('Data Index')
        plt.ylabel('Acceleration')
        plt.legend()

    plt.tight_layout()
    plt.show()


def read_quest_csv(dataFolder, subject):
    path = f"{dataFolder}/{subject}/{subject}_quest.csv"
    ground_truths = []
    stress_pos = 0
    panas_len = 0
    
    with open(path) as f:
        next(f)
        quest = csv.reader(f, delimiter=";")
        for i, row in enumerate(quest):
            if i in [3, 9, 15, 21]: continue
            elif i < 3: #conditions
                values = row[1:6]
                if 'TSST' in values:
                    stress_pos = values.index('TSST')
                ground_truths.append(values)
                
                if i == 2:
                    ground_truths = [list(x) for x in zip(*ground_truths)]
            elif i < 9: #PANAS all
                # print(row[1:])
                ground_truths[i-4].extend(row[1:])
                panas_len = len(row[1:])
            elif i < 15: #STAI 6
                # print(row[1:7])
                ground_truths[i-10].extend(row[1:7])
            elif i < 21: #DIM / SAM 2
                # print(row[1:3])
                ground_truths[i-16].extend(row[1:3])
            else: #SSSQ 6 (only for TSST/stress)
                # print(row[1:7])
                ground_truths[stress_pos].extend(row[1:7])
    
    cols = ['condition', 'start', 'end'] + [f"panas_{i+1}" for i in range(panas_len)] + [f"stai_{i+1}" for i in range(6)] + [f"sam_{i+1}" for i in range(2)] + [f"sssq_{i+1}" for i in range(6)]
    ground_truths = pd.DataFrame(ground_truths, columns=cols)
    return ground_truths