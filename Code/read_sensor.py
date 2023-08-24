import pandas as pd
import numpy as np

import math
from zipfile import ZipFile
import os
import csv

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import resample
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# read data
def read_pkl(dataFolder, subject, type='both'):
    # pkl file is already synchronized
    # only need to realign data rows by up / downsampling
    path = f"{dataFolder}/{subject}/{subject}.pkl"
    sync = pd.read_pickle(path) 
    
    # acc at 1/64g
    sync['signal']['wrist']['ACC'] = sync['signal']['wrist']['ACC'].astype(float) * (1/64)
    
    if type == 'both': # realign every feature to 32hz
        target_sampling_rate=32
        # ------------------------------
        # * perform chest and wrist realignement first, as accelerometer data provides the timing information that can be used as a reference for synchronizing multimodal.
        # downsample chest data to match with wrist data
        # reason: to cope with lack of memory
        # risk: losing information
        # solution: upsample wrist data instead as chest data has more points to match with
        # -------------------------------
        
        # ACC, ECG, EDA, EMG, Resp, Temp 
        chest_acc = sync['signal']['chest']['ACC']
        original_length = chest_acc.shape[0]
        resampling_factor = target_sampling_rate/700
        target_length = int(original_length * resampling_factor)
        data_types = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
    
        # --------------
        #  realign chest data to match data points to 32hz
        # --------------
        chest_data = {
            data_type: align_and_resample_data(sync, 'chest', original_length, target_length, data_type) for data_type in data_types
        }
    
        # Convert label timestamps to the 32Hz time base
        label_indices = np.linspace(0, len(sync['label']) - 1, target_length).astype(int)
        chest_data['label'] = sync['label'][label_indices]
            
        # --------------
        # since chest and wrist aligned, realign the other features to match 32hz to ensure consistency
        # decimate bvp, interpolate eda and temp
        # it might introduces artifacts
        # could try to resample with different sampling rate.
        # ---------------
        wrist_data = realign_wrist(target_length, target_sampling_rate, sync)
    
        sensor = {'chest_acc_x': chest_data['ACC'][:, 0],
                'chest_acc_y': chest_data['ACC'][:, 1],
                'chest_acc_z': chest_data['ACC'][:, 2],
                'chest_ecg': chest_data['ECG'],
                'chest_emg': chest_data['EMG'],
                'chest_eda': chest_data['EDA'],
                'chest_resp': chest_data['Resp'],
                'chest_temp': chest_data['Temp'],
                'wrist_acc_x': wrist_data['ACC'][:, 0],
                'wrist_acc_y': wrist_data['ACC'][:, 1],
                'wrist_acc_z': wrist_data['ACC'][:, 2],
                'wrist_bvp': wrist_data['BVP'],
                'wrist_eda': wrist_data['EDA'],
                'wrist_temp': wrist_data['Temp'],
                'label': chest_data['label']}
    elif type == 'chest':
        # no need to realign
        target_sampling_rate=700
        
        chest_data = sync['signal']['chest']
        original_length = sync['label'].shape[0]
        chest_data['label'] = sync['label'].reshape(original_length)
        
        sensor = {'chest_acc_x': chest_data['ACC'][:, 0].reshape(original_length).astype(float),
                'chest_acc_y': chest_data['ACC'][:, 1].reshape(original_length).astype(float),
                'chest_acc_z': chest_data['ACC'][:, 2].reshape(original_length).astype(float),
                'chest_ecg': chest_data['ECG'].reshape(original_length).astype(float),
                'chest_emg': chest_data['EMG'].reshape(original_length).astype(float),
                'chest_eda': chest_data['EDA'].reshape(original_length).astype(float),
                'chest_resp': chest_data['Resp'].reshape(original_length).astype(float),
                'chest_temp': chest_data['Temp'].reshape(original_length).astype(float),
                'label': chest_data['label']}
        
    else:
        target_sampling_rate=64
        
        target_length = sync['signal']['wrist']['BVP'].shape[0]
        
        wrist_data = realign_wrist(target_length, target_sampling_rate, sync)
        
        sensor = {'wrist_acc_x': wrist_data['ACC'][:, 0],
                'wrist_acc_y': wrist_data['ACC'][:, 1],
                'wrist_acc_z': wrist_data['ACC'][:, 2],
                'wrist_bvp': wrist_data['BVP'],
                'wrist_eda': wrist_data['EDA'],
                'wrist_temp': wrist_data['Temp'],
                'label': wrist_data['label']}    
        
    sensor = pd.DataFrame.from_dict(sensor)
    
    # cleaning
    sensor = clean_data_in_frequency_domain(sensor, threshold=0.1)
    
    # feature engineering
    sensor = add_dominant_frequency(sensor, target_sampling_rate)

    sensor.insert(loc=0, column='id', value=subject)
    return sensor

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
    
    
    cols = ['label', 'start', 'end'] \
        + [f"panas_{i+1}" for i in range(panas_len)] \
        + [f"stai_{i+1}" for i in range(6)] \
        + [f"sam_{i+1}" for i in range(2)] \
        + [f"sssq_{i+1}" for i in range(6)]
    ground_truths = pd.DataFrame(ground_truths, columns=cols)
    
    # feature engineering
    ground_truths = get_time_difference(ground_truths)

    # condition - label
    # Base - 1
    # TSST - 2
    # Fun - 3
    # Medi 1 - 4
    # Medi 2 - 4
    # Assuming Medi 1 and 2 both match tp Meditation label in Synchronised data
    ground_truths["label"] = ground_truths["label"].map({
        "Base": 1, 
        "TSST": 2,
        "Fun": 3,
        'Medi 1': 4,
        "Medi 2": 4})
    
    return ground_truths

# realigning / synchronizing data
def resample_data(data, target_length):
    return resample(data, target_length, axis=0)

def interpolate_data(time_indices, data, fs):
    return np.interp(time_indices, np.arange(len(data)) / fs, data)

def interpolate2d_data(time_indices, data, fs, target_length, channels=3):
    interpolated_data = np.empty([target_length, channels])
    for channel in range(channels):
        interpolated_data[:, channel] = np.interp(
            time_indices, np.arange(len(data)) / fs, data[:, channel]
        )
    # print(interpolated_data.shape)
    return interpolated_data

def align_and_resample_data(sync_data, device, original_length, target_length, data_type=None, fs=None, time_indices=None):
    if data_type:
        data = sync_data['signal'][device][data_type].astype(float)
    else:
        data = sync_data['label']
        
    if data_type != 'ACC':
        data = data.reshape(original_length)
    if fs:
        aligned_data = interpolate_data(time_indices, data, fs) if data_type != 'ACC' else interpolate2d_data(time_indices, data, fs, target_length)
    else:
        aligned_data = resample_data(data, target_length)
    return aligned_data

def realign_wrist(target_length, target_sampling_rate, sync):
    time_indices = np.arange(target_length) / target_sampling_rate
    
    original_length = {
        'ACC': sync['signal']['wrist']['ACC'].shape[0],
        'BVP': len(sync['signal']['wrist']['BVP']),
        'EDA': len(sync['signal']['wrist']['EDA']),
        'Temp': len(sync['signal']['wrist']['TEMP'])
    }
    hz_dict = {
        'ACC': 32,
        'BVP': 64,
        'EDA': 4,
        'Temp': 4
    }
    data_types = ['ACC', 'BVP', 'EDA', 'Temp']
    
    label_indices = np.linspace(0, len(sync['label']) - 1, target_length).astype(int)    
    wrist_data = {'label': sync['label'][label_indices]}
    
    for data_type in data_types:
        if original_length[data_type] > target_length:
            wrist_data[data_type] = align_and_resample_data(sync, 'wrist', original_length[data_type], target_length, data_type.upper()) 
        elif original_length[data_type] < target_length:
            wrist_data[data_type] = align_and_resample_data(sync, 'wrist', original_length[data_type], target_length, data_type.upper(),
                                                            hz_dict[data_type], time_indices)
        else: 
            if data_type != 'ACC':
                wrist_data[data_type] = sync['signal']['wrist'][data_type.upper()].astype(float).reshape(original_length[data_type])
            else: 
                wrist_data[data_type] = sync['signal']['wrist'][data_type.upper()]
    
            
    # assert all(len(data) == target_length for data in wrist_data.values())
    
    return wrist_data

# noise cleaning reference:
# https://towardsdatascience.com/clean-up-data-noise-with-fourier-transform-in-python-7480252fd9c9
def convert_to_frequency_domain(signal):
    return np.abs(rfft(signal)) 

def detect_noise_frequency_domain(frequency_domain_signal, threshold=0.1):
    max_amplitude = np.max(frequency_domain_signal)
    noisy_frequencies = np.where(frequency_domain_signal > max_amplitude * threshold)[0]
    return noisy_frequencies

def clean_data_in_frequency_domain(df, threshold=0.1):
    cleaned_df = df.copy()

    for column in df.columns:
        if column not in ['label']:
            time_domain_signal = df[column].values
            frequency_domain_signal = convert_to_frequency_domain(time_domain_signal)
            noisy_frequencies = detect_noise_frequency_domain(frequency_domain_signal, threshold)
            cleaned_df[column][noisy_frequencies] = np.nan
    
    cleaned_df = cleaned_df.interpolate()
    
    if 'chest_temp' in df.columns:
        cleaned_df = cleaned_df[cleaned_df['chest_temp']>0]
    
    return cleaned_df

# normalize scaling
def zscore_scaling(data):
    scaler = StandardScaler()

    normalized_df = scaler.fit_transform(data)
    
    return normalized_df

def minmax_scaling(data):
    scaler = MinMaxScaler()

    normalized_df = scaler.fit_transform(data)
    
    return normalized_df

def normalize(data, type):
    min_max_columns = ['chest_acc_x', 'chest_acc_y', 'chest_acc_z', 
                            'wrist_acc_x', 'wrist_acc_y', 'wrist_acc_z',
                            'chest_eda', 'wrist_eda']
    standard_columns = ['wrist_bvp', 'chest_resp', 'chest_temp', 'wrist_temp', 'chest_emg', 'chest_ecg']
    if type != 'both':
        min_max_columns = [col for col in min_max_columns if type in col]
        standard_columns = [col for col in standard_columns if type in col]
        
    data[min_max_columns] = minmax_scaling(data[min_max_columns])
    data[standard_columns] = zscore_scaling(data[standard_columns])
    
    return data

# feature engineering:
# -- time difference --
def convert_to_time_format(time_str):
    if '.' in time_str:
        minutes, seconds = map(int, time_str.split('.'))
    else:
        minutes = int(time_str)
        seconds = 0
    hours = minutes // 60
    minutes %= 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def get_time_difference(data):
    for col in ['start', 'end']:
        data[col] = data[col].apply(convert_to_time_format)
        data[col] = pd.to_timedelta(data[col])
    data['time_difference'] = (data['end'] - data['start']).apply(lambda x: f"{int(x.total_seconds() // 60)}.{int(x.total_seconds() % 60):02}")
    data = data.drop(columns=['start', 'end'])  
    
    return data

# -- get dominant frequency from FFT --
# looking at peaks of rfft
# often associated with repeating patterns or periodic behavior in the original signal.
def get_dominant_frequency(signal, sampling_rate):
    fft_result = rfft(signal)
    freqs = rfftfreq(len(signal), d=1/sampling_rate)
    peaks, _ = find_peaks(abs(fft_result))
    # error handling
    if len(peaks) > 0:
        dominant_peak = peaks[np.argmax(fft_result[peaks])]
        dominant_frequency = freqs[dominant_peak]
        return dominant_frequency
    else:
        return None

def add_dominant_frequency(df, target_sampling_rate):
    grouped = df.groupby(['label'])
    features = df.columns.values.tolist()
    features.remove('label')
    dominant_freqs = {feat: [] for feat in features}

    for _, group_data in grouped:
        for feat in features:
            data = group_data[feat].values
            
            # Compute dominant frequency for the feature
            dominant_freq = get_dominant_frequency(data, target_sampling_rate)
            dominant_freqs[feat].append(dominant_freq)

    # Create a new DataFrame to store the computed features
    dominant_freq_df = pd.DataFrame({'label': grouped.groups.keys()})
    for feat in features:
        dominant_freq_df[f'{feat}_dominant_freq'] = dominant_freqs[feat]

    # Merge the computed features back into the original DataFrame
    df = df.merge(dominant_freq_df, on=['label'], how='left')
    
    return df

#  EDA / Visualisation
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
   
def boxplot(data, title):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    
    data.boxplot(column=['chest_acc_x', 'chest_acc_y', 'chest_acc_z'],
                 ax=axes[0,0])
    data.boxplot(column=['wrist_acc_x', 'wrist_acc_y', 'wrist_acc_z'],
                 ax=axes[0,1])
    data.boxplot(column=['chest_ecg', 'chest_emg', 'chest_resp'],
                 ax=axes[0,2])
    data.boxplot(column=['wrist_bvp'],
                 ax=axes[1,0])
    data.boxplot(column=['chest_eda', 'wrist_eda'],
                 ax=axes[1,1])
    data.boxplot(column=['chest_temp', 'wrist_temp'],
                 ax=axes[1,2])

    plt.tight_layout() 
    plt.savefig(f'./EDA/{title}.png')
    plt.show()

# main function   
def full_data_groundtruth(dataFolder, subjects, type='both', questionnaires=False):
    all_subjects = pd.DataFrame()
    
    # error handling
    if type not in ['both', 'chest', 'wrist']: 
        return Exception("Sorry, only term in ['both', 'chest', 'wrist']")
    
    for subject in subjects:
        synch_data = read_pkl(dataFolder, subject, type)
        # encode ground truth - study protocol as class label
        # keep only 1-4
        synch_data = synch_data[synch_data['label'].isin([1, 2, 3, 4])]
        groundtruth = read_quest_csv(dataFolder, subject)
        # break
        
        # if questionnaires self-reports needed
        if not questionnaires:     
            groundtruth = groundtruth[['label', 'time_difference']]
            
        per_subject = synch_data.join(groundtruth, lsuffix='_pkl', rsuffix='_quest')
        per_subject = per_subject.drop(columns=['label_quest']).\
                rename(columns={"label_pkl": "label"}).\
                set_index('id')
                       
        if not all_subjects.empty:
            all_subjects = pd.concat([all_subjects, per_subject])
        else:
            all_subjects = per_subject
    
    
    # boxplot(all_subjects, "after_FFT_cleaning")

    
    # ----------
    # z-score normalization assumes a normal distribution but range varies
    # minmax sensitive to outliers, but gives specific range (0-1)
    # -----------
    all_subjects = normalize(all_subjects, type)

    # boxplot(all_subjects, 'normalised_sensor_data')
    # sns.countplot(data=all_subjects, x="label")
    # plt.xticks([0, 1, 2, 3], ['Baseline', 'Stress', 'Amusement', 'Meditation'])
    # plt.title('Mood label distribution')
    # plt.savefig('./EDA/labels_distribution.png')
    # plt.show()
    
    label = all_subjects.pop('label')
    all_subjects['label'] = label.values.tolist()
    return all_subjects
 
# --------------------
# not necessary to read respiban and empatica, as they must be synchronized with labels
# pkl is already providing synchronized data. 
# --------------------
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

