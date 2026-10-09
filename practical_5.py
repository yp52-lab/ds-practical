import math

import pandas as pd


def calculate_entropy(text):
    """Calculate the Shannon entropy (randomness score) of a text string."""
    if not text or pd.isna(text):
        return 0.0

    text_str = str(text)
    probabilities = [text_str.count(char) / len(text_str) for char in set(text_str)]
    return -sum(probability * math.log2(probability) for probability in probabilities)


def generate_features(input_file):
    print("--- PRACTICAL 5: Feature Engineering for Anomaly Detection ---\n")

    print(f"Loading '{input_file}'...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Total Log Records Loaded: {len(df):,}")

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = df.sort_values(by=["ip_address", "date_time"]).reset_index(drop=True)

    print("\n[Feature 1/5] Calculating total requests per IP address...")
    df["ip_request_count"] = df["ip_address"].map(df["ip_address"].value_counts())

    print("[Feature 2/5] Calculating time gap between consecutive requests per IP...")
    df["time_diff_seconds"] = (
        df.groupby("ip_address")["date_time"].diff().dt.total_seconds().fillna(0.0)
    )

    print("[Feature 3/5] Calculating status code frequency (404 / Error counts)...")
    df["is_404"] = (df["status_code"] == 404).astype(int) if "status_code" in df else 0
    df["status_code_404_count"] = df.groupby("ip_address")["is_404"].transform("sum")
    df.drop(columns=["is_404"], inplace=True)

    print("[Feature 4/5] Parsing User-Agents (Bots vs Real Browsers)...")
    bot_keywords = r"(?:bot|crawler|spider|scraper|sqlmap|nikto|gobuster|curl|python-urllib|nmap)"
    df["is_bot_agent"] = (
        df["user_agent"].astype(str).str.contains(bot_keywords, regex=True, case=False).astype(int)
    )

    print("[Feature 5/5] Calculating URL Entropy (Detecting suspicious randomness)...")
    df["url_entropy"] = df["request_resource"].astype(str).apply(calculate_entropy)

    feature_columns = [
        "ip_address", "ip_request_count", "time_diff_seconds", "status_code_404_count",
        "is_bot_agent", "url_entropy", "label",
    ]
    print("\n--- Summary of Engineered Features (First 5 Rows) ---")
    print(df[feature_columns].head())

    print("\n--- Feature Statistics ---")
    print(df[["ip_request_count", "time_diff_seconds", "status_code_404_count", "is_bot_agent", "url_entropy"]].describe())

    output_file = "engineered_features.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved feature-engineered dataset to '{output_file}'!")


if __name__ == "__main__":
    generate_features("labeled_logs.csv")
