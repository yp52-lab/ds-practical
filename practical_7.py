import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib
if 'DISPLAY' not in os.environ or '--headless' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from path_utils import resolve_input_file, resolve_export_file, BASE_DIR

def is_internal_ip(ip_str):
    """Check if an IP address belongs to standard private/internal IP ranges (RFC 1918 / Loopback)."""
    if pd.isna(ip_str):
        return False
    ip_str = str(ip_str).strip()
    pattern = r'^(?:127\.|10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[0-1])\.|localhost|::1)'
    return bool(re.match(pattern, ip_str))

def wrangle_data(input_file="engineered_features.csv", balanced_file="train_dataset.csv"):
    print("=" * 60)
    print("PRACTICAL 7: Data Wrangling & Aggregated Analysis")
    print("=" * 60)
    
    resolved_input = resolve_input_file(input_file)
    
    # 1. Load dataset
    print(f"[*] Loading dataset from '{resolved_input}'...")
    try:
        df = pd.read_csv(resolved_input, low_memory=False)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Falling back to labeled_logs.csv...")
        try:
            fallback_path = resolve_input_file("labeled_logs.csv")
            df = pd.read_csv(fallback_path, low_memory=False)
        except FileNotFoundError:
            print("Error: No labeled dataset found. Please run Practical 4 / 5 first.")
            sys.exit(1)

    initial_count = len(df)
    print(f"[+] Total Initial Log Records Loaded: {initial_count:,}")

    # ── Normalise column names to match what this practical expects ──────
    # The CSV uses 'date_time', 'ip_address', 'request_resource' — remap them.
    if 'timestamp' not in df.columns:
        ts_src = next((c for c in ['date_time', 'timestamp_dt'] if c in df.columns), None)
        df['timestamp'] = pd.to_datetime(df[ts_src], errors='coerce') if ts_src else pd.NaT
    if 'ip' not in df.columns:
        ip_src = next((c for c in ['ip_address', 'additional_ip'] if c in df.columns), None)
        df['ip'] = df[ip_src].astype(str) if ip_src else 'unknown'
    if 'request_type' not in df.columns:
        # Derive HTTP method from request_resource (format: 'GET /path' or just '/path')
        url_src = next((c for c in ['request_resource', 'request_data', 'url_path'] if c in df.columns), None)
        if url_src:
            df['request_type'] = df[url_src].astype(str).str.extract(r'^(GET|POST|PUT|DELETE|HEAD|OPTIONS)', expand=False).fillna('GET')
        else:
            df['request_type'] = 'GET'

    # Ensure timestamp is datetime type
    print("\n[*] Converting timestamps to datetime format...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # 2. Filter out Bots and Internal IPs
    print("\n[*] Step 1: Filtering out Internal IPs and Bot Traffic...")
    
    # Identify internal IPs
    is_internal = df['ip'].apply(is_internal_ip)
    internal_ip_count = is_internal.sum()
    
    # Identify bot traffic
    if 'is_bot_agent' in df.columns:
        is_bot = df['is_bot_agent'] == 1
    else:
        bot_keywords = r"(?:bot|crawler|spider|scraper|sqlmap|nikto|gobuster|dirbuster|curl|python-urllib|nmap)"
        is_bot = df['user_agent'].astype(str).str.contains(bot_keywords, regex=True, case=False)
        
    bot_count = is_bot.sum()
    
    # Combined filter (Internal OR Bot)
    excluded_mask = is_internal | is_bot
    excluded_count = excluded_mask.sum()
    
    clean_df = df[~excluded_mask].copy().reset_index(drop=True)
    clean_count = len(clean_df)

    filter_summary = [
        ["Total Raw Records", f"{initial_count:,}", "100.00%"],
        ["Internal IP Records Removed", f"{internal_ip_count:,}", f"{(internal_ip_count / initial_count) * 100:.2f}%"],
        ["Bot Traffic Records Identified", f"{bot_count:,}", f"{(bot_count / initial_count) * 100:.2f}%"],
        ["Total Excluded Records (Internal + Bots)", f"{excluded_count:,}", f"{(excluded_count / initial_count) * 100:.2f}%"],
        ["Cleaned / Filtered Records Remaining", f"{clean_count:,}", f"{(clean_count / initial_count) * 100:.2f}%"]
    ]
    print(tabulate(filter_summary, headers=["Metric / Category", "Record Count", "Percentage"], tablefmt="fancy_grid"))

    # 3. Group by IP to Analyze Frequency of Attacks
    print("\n[*] Step 2: Grouping by IP Address to Analyze Attack Frequency...")
    
    # Aggregate requests per IP
    ip_grouped = df.groupby('ip').agg(
        total_requests=('label', 'count'),
        attack_requests=('label', lambda s: (s != 'benign').sum()),
        brute_force_count=('label', lambda s: (s == 'brute_force').sum()),
        sqli_count=('label', lambda s: (s == 'sqli').sum()),
        path_traversal_count=('label', lambda s: (s == 'path_traversal').sum()),
        benign_count=('label', lambda s: (s == 'benign').sum())
    ).reset_index()

    ip_grouped['attack_ratio_%'] = (ip_grouped['attack_requests'] / ip_grouped['total_requests']) * 100
    top_attackers = ip_grouped.sort_values(by='attack_requests', ascending=False).head(15)

    attacker_table = []
    for _, row in top_attackers.iterrows():
        attacker_table.append([
            row['ip'],
            f"{int(row['total_requests']):,}",
            f"{int(row['attack_requests']):,}",
            f"{row['attack_ratio_%']:.1f}%",
            f"{int(row['brute_force_count']):,}",
            f"{int(row['sqli_count']):,}",
            f"{int(row['path_traversal_count']):,}"
        ])
    
    print("\n--- TOP 15 MOST AGGRESSIVE ATTACKING IP ADDRESSES ---")
    print(tabulate(attacker_table, headers=["Source IP", "Total Req", "Attack Req", "Attack Ratio", "BruteForce", "SQLi", "PathTraversal"], tablefmt="fancy_grid"))

    # 4. Time-Series Resampling (Hourly & Daily Traffic)
    print("\n[*] Step 3: Performing Time-Series Resampling (Hourly & Daily Traffic Analysis)...")
    
    # Set timestamp as index for time-series resampling
    ts_df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()

    # Hourly Resampling
    hourly_resampled = ts_df.groupby([pd.Grouper(freq='h'), 'label']).size().unstack(fill_value=0)
    hourly_resampled['Total_Requests'] = hourly_resampled.sum(axis=1)

    print("\n--- HOURLY TRAFFIC RESAMPLING (FIRST 10 HOURLY PERIODS) ---")
    hourly_summary = []
    for ts, row in hourly_resampled.head(10).iterrows():
        hourly_summary.append([
            ts.strftime('%Y-%m-%d %H:%M'),
            f"{int(row.get('Total_Requests', 0)):,}",
            f"{int(row.get('benign', 0)):,}",
            f"{int(row.get('brute_force', 0)):,}",
            f"{int(row.get('sqli', 0)):,}",
            f"{int(row.get('path_traversal', 0)):,}"
        ])
    print(tabulate(hourly_summary, headers=["Timestamp (Hour)", "Total", "Benign", "Brute Force", "SQLi", "Path Traversal"], tablefmt="fancy_grid"))

    # Daily Resampling
    daily_resampled = ts_df.groupby([pd.Grouper(freq='D'), 'label']).size().unstack(fill_value=0)
    daily_resampled['Total_Requests'] = daily_resampled.sum(axis=1)
    
    print("\n--- DAILY TRAFFIC SUMMARY ---")
    daily_summary = []
    for ts, row in daily_resampled.iterrows():
        daily_summary.append([
            ts.strftime('%Y-%m-%d'),
            f"{int(row.get('Total_Requests', 0)):,}",
            f"{int(row.get('benign', 0)):,}",
            f"{int(row.get('brute_force', 0)):,}",
            f"{int(row.get('sqli', 0)):,}",
            f"{int(row.get('path_traversal', 0)):,}"
        ])
    print(tabulate(daily_summary, headers=["Date", "Total", "Benign", "Brute Force", "SQLi", "Path Traversal"], tablefmt="fancy_grid"))

    # 5. Pivot Tables: Request Type by Label
    print("\n[*] Step 4: Constructing Pivot Table (Request Method by Attack Label)...")
    
    df['request_type_clean'] = df['request_type'].fillna('UNKNOWN').astype(str).str.upper()
    pivot_req_label = pd.crosstab(
        df['request_type_clean'],
        df['label'],
        margins=True,
        margins_name="Total"
    )
    
    print("\n--- PIVOT TABLE: REQUEST TYPE vs ATTACK LABEL (COUNTS) ---")
    print(tabulate(pivot_req_label, headers='keys', tablefmt='fancy_grid'))

    # Percentage Pivot Table
    pivot_pct = pd.crosstab(
        df['request_type_clean'],
        df['label'],
        normalize='index'
    ) * 100

    print("\n--- PIVOT TABLE: REQUEST TYPE vs ATTACK LABEL (ROW PERCENTAGES %) ---")
    pivot_pct_formatted = pivot_pct.map(lambda x: f"{x:.2f}%")
    print(tabulate(pivot_pct_formatted, headers='keys', tablefmt='fancy_grid'))

    # 6. Analysis on Balanced Training Set
    if os.path.exists(balanced_file):
        print(f"\n[*] Step 5: Validating Aggregated Structure on Balanced Training Set '{balanced_file}'...")
        bal_df = pd.read_csv(balanced_file)
        bal_pivot = bal_df.groupby('label').agg(
            sample_count=('ip_request_count', 'count'),
            avg_ip_request_count=('ip_request_count', 'mean'),
            avg_time_diff_sec=('time_diff_seconds', 'mean'),
            avg_404_count=('status_code_404_count', 'mean'),
            bot_agent_ratio=('is_bot_agent', 'mean'),
            avg_url_entropy=('url_entropy', 'mean')
        ).reset_index()
        
        bal_summary = []
        for _, r in bal_pivot.iterrows():
            bal_summary.append([
                r['label'].upper(),
                f"{int(r['sample_count']):,}",
                f"{r['avg_ip_request_count']:.2f}",
                f"{r['avg_time_diff_sec']:.4f}s",
                f"{r['avg_404_count']:.2f}",
                f"{r['bot_agent_ratio'] * 100:.1f}%",
                f"{r['avg_url_entropy']:.4f}"
            ])
        print("\n--- BALANCED DATASET FEATURE AGGREGATION BY CLASS ---")
        print(tabulate(bal_summary, headers=["Class", "Samples", "Avg Req Count", "Avg Time Diff", "Avg 404 Count", "Bot Agent %", "Avg URL Entropy"], tablefmt="fancy_grid"))

    # 7. Export Visualizations
    plots_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8')

    # Plot 1: Hourly Time-Series Resampling Plot
    print("\n[*] Exporting plot_p7_hourly_traffic.png...")
    plt.figure(figsize=(14, 6))
    plot_cols = [c for c in ['benign', 'brute_force', 'sqli', 'path_traversal'] if c in hourly_resampled.columns]
    for col in plot_cols:
        plt.plot(hourly_resampled.index, hourly_resampled[col], label=col.replace('_', ' ').title(), linewidth=2)
        
    plt.title("Hourly Traffic Resampling & Attack Category Trends over Time", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Timeline (Hourly Resampled)", fontsize=12)
    plt.ylabel("Number of Requests (Log Scale)", fontsize=12)
    plt.yscale('log')
    plt.legend(title="Traffic / Attack Class", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot_p7_hourly_traffic.png"), dpi=300)
    
    is_interactive = sys.stdin.isatty() and '--headless' not in sys.argv
    if is_interactive:
        try:
            print("    -> Close the plot window to continue...")
            plt.show()
        except Exception:
            pass
    plt.close()

    # Plot 2: Heatmap of Pivot Table (Request Method vs Attack Label)
    print("[*] Exporting plot_p7_pivot_request_by_label.png...")
    plt.figure(figsize=(10, 6))
    pivot_no_total = pivot_req_label.drop(index='Total', errors='ignore').drop(columns='Total', errors='ignore')
    sns.heatmap(pivot_no_total, annot=True, fmt=',d', cmap='YlGnBu', cbar_kws={'label': 'Request Count'})
    plt.title("Pivot Table Heatmap:\nRequest Method vs Security Classification Label", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Assigned Label", fontsize=11)
    plt.ylabel("HTTP Request Method", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot_p7_pivot_request_by_label.png"), dpi=300)
    
    if is_interactive:
        try:
            print("    -> Close the plot window to continue...")
            plt.show()
        except Exception:
            pass
    plt.close()

    # Plot 3: Top 10 Attacking IPs
    print("[*] Exporting plot_p7_top_attacking_ips.png...")
    plt.figure(figsize=(12, 6))
    top_10 = top_attackers.head(10)
    sns.barplot(data=top_10, x='attack_requests', y='ip', hue='ip', palette='magma', legend=False)
    plt.title("Top 10 Most Aggressive Attacking Source IP Addresses", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Total Attack Requests", fontsize=12)
    plt.ylabel("Source IP Address", fontsize=12)
    
    for i, val in enumerate(top_10['attack_requests']):
        plt.text(val + (val * 0.01), i, f" {int(val):,}", va='center', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot_p7_top_attacking_ips.png"), dpi=300)
    
    if is_interactive:
        try:
            print("    -> Close the plot window to continue...")
            plt.show()
        except Exception:
            pass
    plt.close()

    print("[+] Visualizations exported successfully to './plots/'!")
    print("=" * 60)

if __name__ == "__main__":
    wrangle_data()
