import os
import sys
import pandas as pd
import numpy as np
import matplotlib
if 'DISPLAY' not in os.environ or '--headless' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from path_utils import resolve_input_file, BASE_DIR

def perform_eda_visualizations(input_file="engineered_features.csv"):
    print("=" * 60)
    print("PRACTICAL 8: Data Visualization & Exploratory Data Analysis (EDA)")
    print("=" * 60)
    
    resolved_input = resolve_input_file(input_file)
    
    # 1. Load Dataset
    print(f"[*] Loading dataset from '{resolved_input}'...")
    try:
        df = pd.read_csv(resolved_input, low_memory=False)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Falling back to labeled_logs.csv...")
        try:
            fallback_input = resolve_input_file("labeled_logs.csv")
            df = pd.read_csv(fallback_input, low_memory=False)
        except FileNotFoundError:
            print("Error: No dataset found. Please run Practical 4/5 first.")
            sys.exit(1)

    print(f"[+] Total Log Records Loaded: {len(df):,}")

    # ── Normalise column names ──────────────────────────────────────
    # The CSV uses 'date_time', 'ip_address' — remap to standard names.
    if 'timestamp' not in df.columns:
        ts_src = next((c for c in ['date_time', 'timestamp_dt'] if c in df.columns), None)
        df['timestamp'] = pd.to_datetime(df[ts_src], errors='coerce') if ts_src else pd.NaT
    if 'ip' not in df.columns:
        ip_src = next((c for c in ['ip_address', 'additional_ip'] if c in df.columns), None)
        df['ip'] = df[ip_src].astype(str) if ip_src else 'unknown'
    if 'request_type' not in df.columns:
        url_src = next((c for c in ['request_resource', 'request_data'] if c in df.columns), None)
        if url_src:
            df['request_type'] = df[url_src].astype(str).str.extract(
                r'^(GET|POST|PUT|DELETE|HEAD|OPTIONS)', expand=False).fillna('GET')
        else:
            df['request_type'] = 'GET'

    # Ensure timestamp is datetime
    print("[*] Parsing timestamp field...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Setup plots directory and style
    plots_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8')
    is_interactive = sys.stdin.isatty() and '--headless' not in sys.argv

    # -------------------------------------------------------------
    # Visualization 1: Requests per Hour (Time-Series)
    # -------------------------------------------------------------
    print("\n[*] 1/5: Exporting plot_p8_requests_per_hour.png...")
    ts_df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
    hourly_traffic = ts_df.resample('h').size().reset_index(name='request_count')

    plt.figure(figsize=(14, 6))
    plt.plot(hourly_traffic['timestamp'], hourly_traffic['request_count'], color='#1f77b4', linewidth=1.5)
    plt.title("Total Log Requests per Hour (Traffic Volume)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Timeline (Hourly Resampled)", fontsize=12)
    plt.ylabel("Number of Requests (Log Scale)", fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot1_path = os.path.join(plots_dir, "plot_p8_requests_per_hour.png")
    plt.savefig(plot1_path, dpi=300)
    print(f"    [+] Saved to '{plot1_path}'!")
    if is_interactive:
        try:
            print("    -> Close the plot window to continue...")
            plt.show()
        except Exception:
            pass
    plt.close()

    # -------------------------------------------------------------
    # Visualization 2: Top 10 Attacking IPs
    # -------------------------------------------------------------
    print("\n[*] 2/5: Exporting plot_p8_top10_attacking_ips.png...")
    attack_df = df[df['label'] != 'benign']
    top_attacking_ips = attack_df['ip'].value_counts().head(10).reset_index()
    top_attacking_ips.columns = ['ip', 'attack_count']

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_attacking_ips, x='attack_count', y='ip', hue='ip', palette='flare', legend=False)
    plt.title("Top 10 Most Active Attacking Source IP Addresses", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Total Attack Request Volume", fontsize=12)
    plt.ylabel("Source IP Address", fontsize=12)
    
    for i, val in enumerate(top_attacking_ips['attack_count']):
        plt.text(val + (val * 0.01), i, f" {int(val):,}", va='center', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    plot2_path = os.path.join(plots_dir, "plot_p8_top10_attacking_ips.png")
    plt.savefig(plot2_path, dpi=300)
    print(f"    [+] Saved to '{plot2_path}'!")
    if is_interactive:
        try:
            print("    -> Close the plot window to continue...")
            plt.show()
        except Exception:
            pass
    plt.close()

    # Print Top 10 table in console
    top10_table = []
    for _, row in top_attacking_ips.iterrows():
        top10_table.append([row['ip'], f"{row['attack_count']:,}"])
    print("\n--- TOP 10 ATTACKING IP ADDRESSES ---")
    print(tabulate(top10_table, headers=["Source IP Address", "Attack Count"], tablefmt="fancy_grid"))

    # -------------------------------------------------------------
    # Visualization 3: Attack Categories Over Time
    # -------------------------------------------------------------
    print("\n[*] 3/5: Exporting plot_p8_attack_categories_over_time.png...")
    hourly_by_label = ts_df.groupby([pd.Grouper(freq='h'), 'label']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(14, 6))
    category_colors = {'benign': '#2ca02c', 'brute_force': '#1f77b4', 'sqli': '#d62728', 'path_traversal': '#9467bd'}
    
    for col in hourly_by_label.columns:
        c_name = col.replace('_', ' ').title()
        c_color = category_colors.get(col, None)
        plt.plot(hourly_by_label.index, hourly_by_label[col], label=c_name, color=c_color, linewidth=1.8, alpha=0.85)

    plt.title("Distribution of Security Attack Categories over Time", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Timeline (Hourly Resampled)", fontsize=12)
    plt.ylabel("Request Count (Log Scale)", fontsize=12)
    plt.yscale('log')
    plt.legend(title="Security Label", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot3_path = os.path.join(plots_dir, "plot_p8_attack_categories_over_time.png")
    plt.savefig(plot3_path, dpi=300)
    print(f"    [+] Saved to '{plot3_path}'!")
    if is_interactive:
        try:
            print("    -> Close the plot window to continue...")
            plt.show()
        except Exception:
            pass
    plt.close()

    # -------------------------------------------------------------
    # Visualization 4: Status Code Distributions
    # -------------------------------------------------------------
    print("\n[*] 4/5: Exporting plot_p8_status_code_distribution.png...")
    if 'status_code' in df.columns:
        status_counts = df['status_code'].value_counts().reset_index()
        status_counts.columns = ['status_code', 'count']
        status_counts['status_code'] = status_counts['status_code'].astype(str)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=status_counts, x='status_code', y='count', hue='status_code', palette='magma', legend=False)
        plt.title("Distribution of HTTP Response Status Codes", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("HTTP Status Code", fontsize=12)
        plt.ylabel("Frequency Count (Log Scale)", fontsize=12)
        plt.yscale('log')

        for i, val in enumerate(status_counts['count']):
            plt.text(i, val * 1.15, f"{int(val):,}", ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        plot4_path = os.path.join(plots_dir, "plot_p8_status_code_distribution.png")
        plt.savefig(plot4_path, dpi=300)
        print(f"    [+] Saved to '{plot4_path}'!")
        if is_interactive:
            try:
                print("    -> Close the plot window to continue...")
                plt.show()
            except Exception:
                pass
        plt.close()

        # Print Status Code table
        sc_table = []
        for _, r in status_counts.iterrows():
            sc_table.append([r['status_code'], f"{r['count']:,}", f"{(r['count']/len(df))*100:.2f}%"])
        print("\n--- HTTP STATUS CODE DISTRIBUTION ---")
        print(tabulate(sc_table, headers=["Status Code", "Count", "Percentage"], tablefmt="fancy_grid"))

    # -------------------------------------------------------------
    # Visualization 5: Heatmap of IP vs Request Types
    # -------------------------------------------------------------
    print("\n[*] 5/5: Exporting plot_p8_heatmap_ip_request_type.png...")
    top_15_ips = df['ip'].value_counts().head(15).index
    df_top_ips = df[df['ip'].isin(top_15_ips)].copy()
    df_top_ips['request_type_clean'] = df_top_ips['request_type'].fillna('UNKNOWN').astype(str).str.upper()

    pivot_ip_req = pd.crosstab(df_top_ips['ip'], df_top_ips['request_type_clean'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_ip_req, annot=True, fmt=',d', cmap='YlGnBu', cbar_kws={'label': 'Request Count'})
    plt.title("Heatmap: Top 15 IP Addresses vs HTTP Request Methods", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("HTTP Request Method", fontsize=11)
    plt.ylabel("Source IP Address", fontsize=11)
    plt.tight_layout()
    plot5_path = os.path.join(plots_dir, "plot_p8_heatmap_ip_request_type.png")
    plt.savefig(plot5_path, dpi=300)
    print(f"    [+] Saved to '{plot5_path}'!")
    if is_interactive:
        try:
            print("    -> Close the plot window to continue...")
            plt.show()
        except Exception:
            pass
    plt.close()

    print("\n[+] All Practical 8 EDA Visualizations successfully exported to './plots/'!")
    print("=" * 60)

if __name__ == "__main__":
    perform_eda_visualizations()
