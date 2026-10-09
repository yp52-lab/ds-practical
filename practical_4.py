import pandas as pd

def label_log_requests(input_file):
    print("--- PRACTICAL 4: Request Labeling & Classification ---\n")
    
    # 1. Load the cleaned dataset from Practical 3
    print(f"Loading '{input_file}'...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Total Rows to Label: {len(df):,}")
    
    # Ensure text columns are string type
    df['request_resource'] = df['request_resource'].fillna('').astype(str)
    df['user_agent'] = df['user_agent'].fillna('').astype(str)

    # Combine relevant text fields into a single search target
    combined_text = df['request_resource'] + " " + df['user_agent']

    # 2. Define standard security attack patterns (non-capturing groups (?:) to avoid warnings)
    sqli_pattern = r"(?:'|\%27)|(?:\-\-)|(?:\bOR\b.*=.*)|(?:\bUNION\b.*SELECT\b)|(?:\bDROP\b.*TABLE\b)"
    path_traversal_pattern = r"(?:\.\./|\.\.\\|/etc/passwd|boot\.ini)"
    bot_scanner_pattern = r"(?:sqlmap|nikto|gobuster|nmap|python-urllib|curl|hydra)"

    # 3. Apply classification rules across combined text fields
    print("\nApplying security rules across resources and user-agent fields...")
    
    is_sqli = combined_text.str.contains(sqli_pattern, regex=True, case=False)
    is_path_traversal = combined_text.str.contains(path_traversal_pattern, regex=True, case=False)
    is_brute_force = df['user_agent'].str.contains(bot_scanner_pattern, regex=True, case=False)

    # Assign labels based on priority
    df['label'] = 'benign'
    df.loc[is_brute_force, 'label'] = 'brute_force'
    df.loc[is_path_traversal, 'label'] = 'path_traversal'
    df.loc[is_sqli, 'label'] = 'sqli'

    # 4. Print Label Distribution Summary
    print("\n--- Label Distribution Summary ---")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  • {label.upper():<15}: {count:,} ({percentage:.2f}%)")

    print("\n--- First 5 Labeled Rows ---")
    print(df[['ip_address', 'request_resource', 'user_agent', 'label']].head())

    # 5. Save output as labeled dataset
    output_file = "labeled_logs.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved labeled dataset to '{output_file}'!")

if __name__ == "__main__":
    label_log_requests("cleaned_logs.csv")