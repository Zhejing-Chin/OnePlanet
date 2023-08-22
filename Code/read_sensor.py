import pandas as pd
import math
from zipfile import ZipFile
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, decimate
from scipy.interpolate import interp1d

# def read_respiban_txt(dataFolder, subject):
#     vcc=3
#     chan_bit=2**16
#     Cmin = 28000 
#     Cmax = 38000
    
#     path = f"{dataFolder}/{subject}/{subject}_respiban.txt"
#     rows = []
#     with open(path) as f:    
#         next(f)
#         next(f)
#         # header = json.loads(f.readline().strip()[2:])
#         next(f)    
#         for line in f:
#             values = line.strip().split()
            
#             ecg = ((float(values[2])/chan_bit-0.5)*vcc)
#             eda = (((float(values[3])/chan_bit)*vcc)/0.12)
#             emg = ((float(values[4])/chan_bit-0.5)*vcc)
#             vout = (float(values[5])*vcc)/(chan_bit-1.) 
#             rntc = ((10**4)*vout)/(vcc-vout) 
#             # assume log base e
#             temp = -1
#             try:
#                 temp = - 273.15 + 1./(1.12764514*(10**(-3)) + 2.34282709*(10**(-4))*math.log(rntc) + 8.77303013*(10**(-8))*(math.log(rntc)**3))
#             except ValueError:
#                 print(values[0], rntc)
#             xyz = [(float(x)-Cmin)/(Cmax-Cmin)*2-1 for x in values[6:9]]
#             resp = (float(values[9]) / chan_bit - 0.5) * 100
            
#             rows.append({'id':subject, 
#             'seq':values[0],
#             'ecg':ecg, 
#             'eda':eda, 
#             'emg':emg, 
#             'temp':temp,
#             'xyz_1':xyz[0],       
#             'xyz_2':xyz[1], 
#             'xyz_3':xyz[2], 
#             'respiration':resp})
        
#     return rows   
        
# def get_respiban(dataFolder, subjects):
#     respiban = pd.DataFrame({'id':'', 
#                              'seq':int(),
#                              'ecg':float(), 
#                              'eda':float(), 
#                              'emg':float(), 
#                              'temp':float(),
#                              'xyz_1':float(),       
#                              'xyz_2':float(), 
#                              'xyz_3':float(), 
#                              'respiration':float()}, index=[])
    
#     for subject in subjects:
#         rows = read_respiban_txt(dataFolder, subject)
        
#         respiban = respiban.append(rows,
#             ignore_index = True)
        
#     return respiban

# --------------------
# not necessary to read empatica, as it must be manually synchronized with respiban to get labels
# pkl is already providing synchronized data
# --------------------
# def unzip_empatica(dataFolder, subject):
#     path = f"{dataFolder}/{subject}"
#     # print(path)
#     check_unzipped = next(os.walk(path))[1]
#     if not f"{subject}_E4_Data" in check_unzipped:
#         zipfile = f"{dataFolder}/{subject}/{subject}_E4_Data.zip"
#     # print(path)
#         with ZipFile(zipfile, 'r') as zObject:
#             zObject.extractall(path+f"/{subject}_E4_Data")
            
# def read_empatica_csv(dataFolder, subject):
#     # make sure folder is unzipped
#     unzip_empatica(dataFolder, subject)
    
#     csv_files = ['ACC', 'BVP', 'EDA', 'TEMP']

def resample_data(data, target_length):
    return resample(data, target_length, axis=0)

def interpolate_data(time_indices, data, fs):
    return np.interp(time_indices, np.arange(len(data)) / fs, data)

def align_and_resample_data(sync_data, device, original_length, target_length, data_type=None, fs=None, time_indices=None):
    if data_type:
        data = sync_data['signal'][device][data_type].astype(float)
    else:
        data = sync_data['label']
        
    if data_type != 'ACC':
        data = data.reshape(original_length)
    if fs:
        aligned_data = interpolate_data(time_indices, data, fs)
    else:
        aligned_data = resample_data(data, target_length)
    return aligned_data

def read_pkl(dataFolder, subject):
    # pkl file is already synchronized
    # only need to realign data rows by up / downsampling
    
    path = f"{dataFolder}/{subject}/{subject}.pkl"
    sync = pd.read_pickle(path) 
    
    # ACC, ECG, EDA, EMG, Resp, Temp 
    resampling_factor = 32/700
    chest_acc = sync['signal']['chest']['ACC']
    original_length = chest_acc.shape[0]
    target_length = int(original_length * resampling_factor)
    # print(original_length, target_length)
    data_types = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
    
    # --------------
    #  realign chest data to match data points to 32hz
    # --------------
    chest_data_32hz = {
        data_type: align_and_resample_data(sync, 'chest', original_length, target_length, data_type) for data_type in data_types
    }
    # Convert label timestamps to the 32Hz time base
    label_indices = np.linspace(0, len(sync['label']) - 1, target_length).astype(int)
    chest_data_32hz['label'] = sync['label'][label_indices]
    assert sum(chest_data_32hz['label']) != 0

    # ACC, BVP, EDA, TEMP
    # acc at 1/64g
    wrist_acc = sync['signal']['wrist']['ACC'].astype(float) * (1/64)

    # ------------------------------
    # * perform chest and wrist realignement first, as accelerometer data provides the timing information that can be used as a reference for synchronizing multimodal.
    # downsample chest data to match with wrist data
    # reason: to cope with lack of memory
    # risk: losing information
    # solution: upsample wrist data instead as chest data has more points to match with
    # -------------------------------
    assert all(len(data) == target_length for data in chest_data_32hz.values()) and \
       len(wrist_acc) == target_length
        
        
    # --------------
    # since chest and wrist aligned, realign the other features to match 32hz to ensure consistency
    # decimate bvp, interpolate eda and temp
    # it might introduces artifacts
    # could try to resample with different sampling rate.
    # ---------------
    target_sampling_rate = 32  # Hz
    time_indices = np.arange(target_length) / target_sampling_rate
    
    length_bvp = len(sync['signal']['wrist']['BVP'])
    length_eda = len(sync['signal']['wrist']['EDA'])
    length_temp = len(sync['signal']['wrist']['TEMP'])
    
    wrist_data_32hz = {
        'BVP': align_and_resample_data(sync, 'wrist', length_bvp, target_length, 'BVP'),
        'EDA': align_and_resample_data(sync, 'wrist', length_eda, target_length, 'EDA', 4, time_indices),
        'Temp': align_and_resample_data(sync, 'wrist', length_temp, target_length, 'TEMP', 4, time_indices),
    }
    assert all(len(data) == target_length for data in wrist_data_32hz.values()) and \
       len(wrist_acc) == target_length
    
    # plot_acceleration(resampled_chest_data.transpose(), acc_wrist.transpose())
    
    # assume label 4 link to both med 1 and med 2 in ground truth
    sensor = {'chest_acc_x': chest_data_32hz['ACC'][:, 0],
             'chest_acc_y': chest_data_32hz['ACC'][:, 1],
             'chest_acc_z': chest_data_32hz['ACC'][:, 2],
             'chest_ecg': chest_data_32hz['ECG'],
             'chest_emg': chest_data_32hz['EMG'],
             'chest_eda': chest_data_32hz['EDA'],
             'chest_resp': chest_data_32hz['Resp'],
             'chest_temp': chest_data_32hz['Temp'],
             'wrist_acc_x': wrist_acc[:, 0],
             'wrist_acc_y': wrist_acc[:, 1],
             'wrist_acc_z': wrist_acc[:, 2],
             'wrist_bvp': wrist_data_32hz['BVP'],
             'wrist_eda': wrist_data_32hz['EDA'],
             'wrist_temp': wrist_data_32hz['Temp'],
             'label': chest_data_32hz['label']}
    sensor = pd.DataFrame.from_dict(sensor)
    sensor.insert(loc=0, column='id', value=subject)
    
    return sensor


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