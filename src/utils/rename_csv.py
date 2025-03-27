import pandas as pd
import argparse

def rename_csv(csv_path):
    df = pd.read_csv(csv_path)
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.replace(".npy", ".nii.gz")

    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename .npy to .nii.gz in CSV")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file")
    
    args = parser.parse_args()
    
    rename_csv(args.csv_path)