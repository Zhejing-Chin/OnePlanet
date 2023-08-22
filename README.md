# OnePlanet

WESAD dataset processing

Link to dataset: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/ 

Problems definition: 
1. To create a data processing pipeline for WESAD sensor data. 
2. The resulting datasets should be in consistent format. (Same column names and types and order)
3. The end output helps in easier analysing and using for building predictive mood model (classification).
4. The processed data should be clear and full with details to provide greater space for feature engineering / extraction.

Assumptions:
1. The class labels are of study protocol (1-4).
    - higher completion of data (no missing values)
    - straightforward to interpret
    - self reports are subjective and might provide noise, higher complexity and influence to final output.

ML cycle:
1. Data processing
2. Data cleaning
3. Data visualisation
4. Feature Engineering
5. Model construction
6. Model evaluation
7. Prediction

