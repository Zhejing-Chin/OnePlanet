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
    - loop through each subject - SX
    - 
     
4. The processed data should be clear and full with details to provide greater space for feature engineering / extraction.



ML cycle:
1. Data processing
2. Data cleaning
3. Data visualisation
4. Feature Engineering
5. Model construction
6. Model evaluation
7. Prediction

