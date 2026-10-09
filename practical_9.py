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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from path_utils import resolve_input_file, BASE_DIR

def train_and_evaluate_models(train_file="train_dataset.csv", test_file="test_dataset.csv"):
    print("=" * 60)
    print("PRACTICAL 9: Building Machine Learning Classifiers for Attack Detection")
    print("=" * 60)

    resolved_train = resolve_input_file(train_file)
    resolved_test = resolve_input_file(test_file)

    # 1. Load Datasets
    print(f"[*] Loading training dataset from '{resolved_train}'...")
    try:
        train_df = pd.read_csv(resolved_train)
    except FileNotFoundError:
        print(f"Error: {train_file} not found. Please run Practical 6 first.")
        sys.exit(1)

    print(f"[*] Loading testing dataset from '{resolved_test}'...")
    try:
        test_df = pd.read_csv(resolved_test)
    except FileNotFoundError:
        print(f"Error: {test_file} not found. Please run Practical 6 first.")
        sys.exit(1)

    feature_cols = ['ip_request_count', 'time_diff_seconds', 'status_code_404_count', 'is_bot_agent', 'url_entropy']
    target_col = 'label'

    # Clean NaNs if any
    train_df = train_df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    test_df = test_df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print(f"[+] Balanced Training Set Size: {len(X_train):,} samples")
    print(f"[+] Testing Set Size: {len(X_test):,} samples")

    labels_order = sorted(y_train.unique())
    display_labels = [l.upper() for l in labels_order]

    # Feature Scaling for models sensitive to scale (Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # Model 1: Logistic Regression Classifier
    # -------------------------------------------------------------
    print("\n[*] Training Model 1: Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    lr_acc = accuracy_score(y_test, y_pred_lr)
    lr_prec = precision_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    lr_rec = recall_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    lr_f1 = f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=labels_order)

    print("\n--- LOGISTIC REGRESSION CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred_lr, target_names=display_labels, digits=4))

    # -------------------------------------------------------------
    # Model 2: Random Forest Classifier
    # -------------------------------------------------------------
    print("\n[*] Training Model 2: Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_prec = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    rf_rec = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    rf_f1 = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=labels_order)

    print("\n--- RANDOM FOREST CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred_rf, target_names=display_labels, digits=4))

    # -------------------------------------------------------------
    # Model Comparison Summary Table
    # -------------------------------------------------------------
    comparison_table = [
        ["Logistic Regression", f"{lr_acc * 100:.2f}%", f"{lr_prec * 100:.2f}%", f"{lr_rec * 100:.2f}%", f"{lr_f1 * 100:.2f}%"],
        ["Random Forest Classifier", f"{rf_acc * 100:.2f}%", f"{rf_prec * 100:.2f}%", f"{rf_rec * 100:.2f}%", f"{rf_f1 * 100:.2f}%"]
    ]
    print("\n--- MODEL PERFORMANCE EVALUATION COMPARISON ---")
    print(tabulate(comparison_table, headers=["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score"], tablefmt="fancy_grid"))

    # Feature Importance Analysis (Random Forest)
    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    print("\n--- RANDOM FOREST FEATURE IMPORTANCE RANKING ---")
    imp_table = []
    for _, r in feat_imp_df.iterrows():
        imp_table.append([r['Feature'], f"{r['Importance'] * 100:.2f}%"])
    print(tabulate(imp_table, headers=["Feature Name", "Importance Weight (%)"], tablefmt="fancy_grid"))

    # -------------------------------------------------------------
    # Export Plots
    # -------------------------------------------------------------
    plots_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8')
    is_interactive = sys.stdin.isatty() and '--headless' not in sys.argv and matplotlib.get_backend().lower() != 'agg'

    # Create comparison DataFrame for plotting
    comp_df = pd.DataFrame([
        {'Model': 'Logistic Regression', 'Accuracy': lr_acc, 'Precision': lr_prec, 'Recall': lr_rec, 'F1-Score': lr_f1},
        {'Model': 'Random Forest Classifier', 'Accuracy': rf_acc, 'Precision': rf_prec, 'Recall': rf_rec, 'F1-Score': rf_f1}
    ])

    # Plot 1: Side-by-Side Confusion Matrices
    print("\n[*] Exporting plot_p9_confusion_matrices.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm_lr, annot=True, fmt=',d', cmap='Blues', ax=axes[0],
                xticklabels=display_labels, yticklabels=display_labels)
    axes[0].set_title(f"Logistic Regression\nAccuracy: {lr_acc*100:.2f}%", fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].tick_params(axis='y', rotation=0)
    axes[0].set_xticklabels(display_labels, rotation=25, ha='right')

    sns.heatmap(cm_rf, annot=True, fmt=',d', cmap='Greens', ax=axes[1],
                xticklabels=display_labels, yticklabels=display_labels)
    axes[1].set_title(f"Random Forest Classifier\nAccuracy: {rf_acc*100:.2f}%", fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].tick_params(axis='y', rotation=0)
    axes[1].set_xticklabels(display_labels, rotation=25, ha='right')

    plt.tight_layout()
    plot1_path = os.path.join(plots_dir, "plot_p9_confusion_matrices.png")
    plt.savefig(plot1_path, dpi=300)
    print(f"[+] Exported confusion matrix plot to '{plot1_path}'!")

    if is_interactive:
        try:
            print("    -> Close the plot window to continue...")
            plt.show()
        except Exception:
            pass
    plt.close()

    # Plot 2: Random Forest Feature Importances
    plt.figure(figsize=(10, 6))
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    sns.barplot(data=feat_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title("Random Forest Feature Importances for Attack Detection", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Gini Importance Score", fontsize=12)
    plt.ylabel("Engineered Feature", fontsize=12)
    plt.tight_layout()
    plot2_path = os.path.join(plots_dir, "plot_p9_feature_importance.png")
    plt.savefig(plot2_path, dpi=300)
    print(f"[+] Exported feature importance plot to '{plot2_path}'!")

    if is_interactive:
        try:
            print("    -> Close the plot window to continue...")
            plt.show()
        except Exception:
            pass
    plt.close()

    # Plot 3: Model Performance Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    comp_df_melted = pd.melt(comp_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                             var_name='Metric', value_name='Score')

    sns.barplot(data=comp_df_melted, x='Metric', y='Score', hue='Model', palette='muted')
    plt.title("Classifier Performance Metric Comparison (Log Attack Detection)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.ylim(0, 1.15)

    for p in plt.gca().patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            plt.gca().annotate(f'{height:.3f}',
                               (p.get_x() + p.get_width() / 2., height + 0.02),
                               ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.legend(title="Model Algorithm", fontsize=10)
    plt.tight_layout()
    plot3_path = os.path.join(plots_dir, "plot_p9_model_comparison.png")
    plt.savefig(plot3_path, dpi=300)
    print(f"    [+] Saved to '{plot3_path}'!")
    if is_interactive:
        try:
            print("    -> Close the plot window to continue...")
            plt.show()
        except Exception:
            pass
    plt.close()

    print("\n[+] Practical 9 Model Training & Evaluation successfully completed!")
    print("=" * 60)

if __name__ == "__main__":
    train_and_evaluate_models()
