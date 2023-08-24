import os
import pandas as pd
import argparse
from read_subject import get_personal_information
from read_sensor import full_data_groundtruth

""" 
Problems definition: 
1. To create a data processing pipeline for WESAD sensor data. 
2. The resulting datasets should be in consistent format. (Same column names and types and order)
3. The end output helps in easier analysing and using for building predictive mood model (classification).
4. The processed data should be clear and full with details to provide greater space for feature engineering / extraction.

Assumptions:
1. The class labels are of study protocol (1-4).
    - higher completion of data
    - straightforward to interpret
    - self reports are subjective and might provide noise, higher complexity and influence to final output.
2. The dataset is class imbalanced, however no resampling is required at this stage as the issue could be mititgated with ML approaches.
    - to stay true with the data. 
    - resampling could be done afterwards if ML approaches did not perform well. 
3. The self-reports are only be treated as an additional feature (could be skipped if not requested)

Git repo: https://github.com/Zhejing-Chin/OnePlanet
Functions in separate files for easier management and code reusabiltiy. 
"""
# python ./Code/main.py --path ./WESAD --type both --output_path ./output/full_data.csv 

def save_csv_to_path(df, output_path):
    path, _ = os.path.split(output_path)
    os.makedirs(path, exist_ok=True)
    df.to_csv(output_path)
    
def main():
    parser = argparse.ArgumentParser(description="Process WESAD sensor data.")
    parser.add_argument("--path", type=str, required=True, help="Path to the data directory (.../WESAD)")
    parser.add_argument("--type", type=str, choices=["both", "chest", "wrist"], default="both", help="Type of sensor data to use")
    parser.add_argument("--questionnaires", action="store_true", help="Include questionnaires data (default: False)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output CSV file ({path}/{name}.csv)")

    args = parser.parse_args()

    subjects = next(os.walk(args.path))[1]

    personal_information = get_personal_information(args.path, subjects)

    sensor_data = full_data_groundtruth(args.path, subjects, type=args.type, questionnaires=args.questionnaires)

    full_data = personal_information.join(sensor_data, how="right", on="id")
    
    # Save the full_data DataFrame to a CSV file
    save_csv_to_path(full_data, args.output_path)
    print(f"Data saved to {args.output_path}")

if __name__ == "__main__":
    main()
