import json
import pandas as pd
import os

def parse_logs(log_file="Logs/logs/cj.log"):
    print(f"Log File: {log_file}")
    
    if not os.path.exists(log_file):
        print(f"Error: {log_file} does not exist.")
        return

    # Load the Raw Log
    with open(log_file, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()
    
    print("Total log records:", len(lines))
    
    if len(lines) > 0:
        # Look at One Record
        print("\n--- Sample Record ---")
        print(lines[0].strip())
        
        # Parse One Line
        try:
            record = json.loads(lines[0])
            print(record)
            print("Number of fields:", len(record))
            for i, value in enumerate(record):
                print(i, ":", value)
        except json.JSONDecodeError:
            pass
            
    # Convert ALL Records
    structured_data = []
    for line in lines:
        try:
            record = json.loads(line)
            if len(record) == 8:
                structured_data.append(record)
        except json.JSONDecodeError:
            continue

    print("\nSuccessfully parsed records:", len(structured_data))
    
    # Create Pandas DataFrame
    columns = [
        "request_resource",
        "request_data",
        "date_time",
        "ip_address",
        "port",
        "user_agent",
        "language",
        "additional_ip"
    ]
    df = pd.DataFrame(structured_data, columns=columns)
    print("Dataset created successfully!")
    
    # Check the Dataset
    print("\nRows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Columns List:", df.columns.tolist())
    
    print("\nDataset Info:")
    df.info()
    
    # Display missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Display unique values in request_resource
    print("\nTop 10 Request Resources:")
    print(df["request_resource"].value_counts().head(10))

    # Save Structured Dataset
    output_file = "structured_logs.csv"
    df.to_csv(output_file, index=False)
    print(f"\nStructured dataset saved successfully to {output_file}")
    
    # Verify the Saved CSV
    check_df = pd.read_csv(output_file, low_memory=False)
    print("Saved rows:", len(check_df))
    print("Saved columns:", len(check_df.columns))

if __name__ == "__main__":
    parse_logs("Logs/logs/cj.log")