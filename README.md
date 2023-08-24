# OnePlanet

## WESAD dataset processing

Link to dataset: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/ 

### Main task: 
* To prepare the WESAD data for predictive mood model.

### Assumptions:
1. The predictive mood model is 1 level multiclass classification.
2. The class labels are of study protocol (1-4).
    - higher completion of data (no missing values)
    - straightforward to interpret
    - self reports are subjective and might provide noise, higher complexity and influence to final output.
3. The goal is to use signals on wearables to predict mood. 
4. Medi 1 and Medi 2 study protocols both pointing to class: Meditation (4)

### Solution: 
1. read_subject.py
    - loop through each subject - SX
    - read SX_readme.txt and create dataframe
    - check for missing values
    - generate EDA for statistical analysis
    - embed "additional notes" as feature engineering 
    - encode categorical columns (yes:1, no:0, etc)
    - normalize age, height, weight to ensure no features dominate the model

2. read_sensor.py
    - full_data_groundtruth()
        - take in path, subject list, type of data (both wearables / chest / wrist only), option to include self-reports answers
        - loop through each subject - SX
            - read_pkl()
            - read_quest_csv()
            - join pkl and quest output
            - normalize signals to appropriate range
            - make sure 'label' is at the very end of dataframe
    - read_pkl()
        - read specific subject's pkl file to get synchronized data
        - standardise the unit of signals
            - if type of data == both
                - realign both data to fit wrist ACC (32Hz)
            - else if == chest
                - no realignment needed
            - else 
                - realign wrist data and labels to wrist BVP (64Hz)
        > interpolate signals to match lower hz to higher hz\\
        > resample signals to match higher hz to lower hz
        - cleaning
            - clean data with fourier transform
            - convert time domain data to frequency domain and clean signals above threshold
            - hard clean rows with chest temperature < 0
        - feature engineering
            - add dominant frequency for each type of signal based on each label
                - to provide information of what's the dominant frequency look like in different mood
    - read_quest_csv()
        - extract only 4 conditions (baseline, stress, fun, meditations) that are relevant
        - cleaning
            - encode conditions to label
        - feature engineering
            - calculate time span for each condition 
3. main.py
    - 

```sh
cd path
python ./Code/main.py --path ./WESAD --type both --output_path ./output/full_data.csv 
```


ML cycle:
1. Data processing
2. Data cleaning
3. Data visualisation
4. Feature Engineering
5. Model construction
6. Model evaluation
7. Prediction

