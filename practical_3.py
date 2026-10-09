import pandas as pd

def clean_and_preprocess_logs(input_file):
    print("--- PRACTICAL 3: Data Cleaning and Preprocessing ---\n")
    
    # 1. Load structured dataset
    print(f"Loading dataset from '{input_file}'...")
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run Practical 2 first.")
        return
        
    initial_count = len(df)
    print(f"Initial Total Rows Loaded: {initial_count:,}")
    
    # 2. Convert date_time column safely
    print("\n[Step 1] Converting 'date_time' column to Datetime format...")
    if 'date_time' in df.columns:
        # Clean up quote marks or whitespace from string timestamps if any
        df['date_time'] = df['date_time'].astype(str).str.replace('"', '').str.strip()
        
        # Try parsing dates with Pandas datetime
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        
        # If any timestamps were invalid, fill them with a fallback datetime instead of dropping
        if df['date_time'].isna().sum() > 0:
            invalid_count = df['date_time'].isna().sum()
            print(f"  ⚠️ Warning: {invalid_count:,} timestamps could not be parsed. Filling with default.")
            df['date_time'] = df['date_time'].fillna(pd.Timestamp('2023-01-08 00:00:00'))

    # 3. Clean string columns and missing values
    print("[Step 2] Cleaning missing values and normalizing text...")
    df.replace(['null', 'NaN', 'nan', 'None', '-'], pd.NA, inplace=True)

    # Normalize text strings
    text_columns = ['request_resource', 'request_data', 'user_agent', 'language', 'additional_ip']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str).str.lower().str.strip()

    if 'ip_address' in df.columns:
        df['ip_address'] = df['ip_address'].fillna('0.0.0.0').astype(str).str.strip()

    # 4. Normalize URL paths (PDF Requirement)
    print("[Step 3] Normalizing URL paths...")
    if 'request_resource' in df.columns:
        # Remove trailing slashes and common extensions like .html
        df['request_resource'] = df['request_resource'].str.rstrip('/')
        df['request_resource'] = df['request_resource'].str.replace(r'\.html$', '', regex=True)

    # 5. Display Summary & Save Cleaned Data
    cleaned_count = len(df)
    print(f"\n✅ Data Cleaning Complete! Retained {cleaned_count:,} out of {initial_count:,} rows.")
    
    print("\n--- Cleaned DataFrame Info ---")
    print(df.info())
    
    print("\n--- First 5 Rows of Cleaned Dataset ---")
    print(df.head())
    
    output_file = "cleaned_logs.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved cleaned dataset to '{output_file}'!")

if __name__ == "__main__":
    clean_and_preprocess_logs("structured_logs.csv")