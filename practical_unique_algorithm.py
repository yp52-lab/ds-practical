"""
Practical — Unique Custom Algorithm
====================================
Algorithm : Adaptive Temporal Threat Profiler (ATTP)
Author    : DS Lab Project

Description
-----------
ATTP is an original, rule-free, statistics-driven anomaly detection algorithm
designed specifically for web server access-log data. It is fundamentally
different from standard classifiers (Logistic Regression, Random Forest, SVM,
KNN, Naive Bayes, DBSCAN, Isolation Forest, etc.).

Core Idea
---------
ATTP operates in two phases:

  Phase 1 — Population Baseline:
    Compute global mean and std for each feature from the full dataset.
    This forms a stable "normal traffic" reference distribution.

  Phase 2 — Streaming Anomaly Detection (row by row, chronologically):
    For each log entry, compute a Dual-Layer Anomaly Score:
      Layer A (Population Z-Score)  : How far is this request from overall traffic?
      Layer B (IP Behavioral Drift) : How far has this IP drifted from ITS OWN
                                      recent history (time-decay weighted profile)?
    The two layers are fused via a weighted arithmetic mean.
    The decision threshold adapts per-IP using Hellinger Distance between the
    IP's current profile and the population baseline.

Three unique components make ATTP novel:
  1. Time-Decay Weighted IP Profiles (online Welford's algorithm with
     exponential decay weights — recent behaviour matters more).
  2. Dual-Layer Fusion (population deviation + IP self-deviation, blended
     with a configurable alpha to handle cold-start gracefully).
  3. Hellinger-Distance Adaptive Threshold (boundary rises for IPs whose
     profile diverges legitimately from population, preventing false positives
     on power-users).
"""

import os
import sys
import math
import argparse
import warnings
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════════════════
#  Helper — Online Weighted Running Statistics (Welford's Algorithm)
# ════════════════════════════════════════════════════════════════════════

class WeightedRunningStats:
    """
    Online computation of weighted mean and variance using an adaptation of
    Welford's one-pass algorithm for non-uniform (time-decay) weights.

    Attributes
    ----------
    w_sum : float   Accumulated weight total.
    mean  : float   Current weighted mean.
    M2    : float   Weighted sum of squared deviations (for variance).
    """
    __slots__ = ('w_sum', 'mean', 'M2')

    def __init__(self):
        self.w_sum = 0.0
        self.mean  = 0.0
        self.M2    = 0.0

    def update(self, value: float, weight: float = 1.0):
        """Incorporate one new (value, weight) observation."""
        self.w_sum += weight
        delta       = value - self.mean
        self.mean  += weight * delta / self.w_sum
        self.M2    += weight * delta * (value - self.mean)

    def decay(self, factor: float):
        """Fade prior observations before incorporating the next one."""
        self.w_sum *= factor
        self.M2 *= factor

    @property
    def variance(self) -> float:
        if self.w_sum < 1e-9:
            return 1.0          # no data yet — return unit variance
        return max(self.M2 / self.w_sum, 1e-6)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


# ════════════════════════════════════════════════════════════════════════
#  ATTP — Adaptive Temporal Threat Profiler
# ════════════════════════════════════════════════════════════════════════

class AdaptiveTemporalThreatProfiler:
    """
    Adaptive Temporal Threat Profiler (ATTP)
    -----------------------------------------
    A novel dual-layer, statistics-driven anomaly detection algorithm for
    web server log data.

    Parameters
    ----------
    decay_lambda      : float  Exponential decay coefficient λ for IP profiles.
                               Higher → older observations lose weight faster.
    anomaly_threshold : float  Base composite z-score cutoff for flagging attacks.
    blend_alpha       : float  [0, 1] Weight on population z-score vs IP z-score.
                               alpha=1.0 → pure population; alpha=0.0 → pure IP.
    feature_weights   : dict   Per-feature importance weights (default provided).
    """

    FEATURE_COLS = [
        'ip_request_count',
        'time_diff_seconds',
        'status_code_404_count',
        'is_bot_agent',
        'url_entropy',
    ]

    DEFAULT_WEIGHTS = {
        'ip_request_count'     : 1.5,   # high-volume IPs → more suspicious
        'time_diff_seconds'    : 1.2,   # very small gaps → automated scanning
        'status_code_404_count': 1.8,   # many 404s → active reconnaissance
        'is_bot_agent'         : 2.0,   # bot user-agent → strongest signal
        'url_entropy'          : 1.3,   # high-entropy URLs → obfuscation
    }

    def __init__(self,
                 decay_lambda: float      = 0.001,
                 anomaly_threshold: float = 2.0,
                 blend_alpha: float       = 0.70,
                 feature_weights: dict    = None):
        self.decay_lambda      = decay_lambda
        self.anomaly_threshold = anomaly_threshold
        self.blend_alpha       = blend_alpha
        self.feature_weights   = feature_weights or self.DEFAULT_WEIGHTS

        # Population baseline (pre-computed in Phase 1)
        self._pop_mean: dict[str, float] = {}
        self._pop_std:  dict[str, float] = {}

        # Per-IP online exponential-decay profiles (Phase 2)
        self._profiles: dict[str, dict[str, WeightedRunningStats]] = {}

        self._predictions: list = []
        self._scores:      list = []

    # ── Phase 1: Pre-compute Population Baseline ─────────────────────────
    def _compute_population_baseline(self, df: pd.DataFrame, available: list):
        """
        Compute global mean and standard deviation for every feature from
        the full dataset BEFORE streaming begins.

        This ensures the population reference is always stable and not
        influenced by the ordering of requests.
        """
        for feat in available:
            vals = df[feat].dropna().astype(float)
            self._pop_mean[feat] = float(vals.mean())
            std = float(vals.std())
            # Enforce a minimum std so z-scores never blow up
            self._pop_std[feat]  = max(std, vals.mean() * 0.05 + 1e-3)

    # ── Scoring Helpers ───────────────────────────────────────────────────
    def _decay_weight(self, row_index: int) -> float:
        """Kept for API compatibility; new observations always have full weight."""
        return 1.0

    def _pop_z(self, feat: str, val: float) -> float:
        """Layer A: z-score of val against the global population distribution."""
        mean = self._pop_mean.get(feat, 0.0)
        std  = self._pop_std.get(feat, 1.0)
        return abs(val - mean) / std

    def _ip_z(self, ip: str, feat: str, val: float) -> float:
        """
        Layer B: z-score of val against the IP's own time-decay weighted
        running profile. Returns 0.0 if the IP profile is too sparse
        (< 3 effective observations) to avoid cold-start false positives.
        """
        profile = self._profiles.get(ip)
        if profile is None:
            return 0.0
        stat = profile.get(feat)
        if stat is None or stat.w_sum < 2.0:
            return 0.0
        ip_std = max(stat.std, 1e-3)
        return abs(val - stat.mean) / ip_std

    def _composite_score(self, ip: str, features: dict) -> float:
        """
        Dual-Layer Weighted Arithmetic Mean Score.

        score = Σ w_i · [α·z_pop_i  +  (1-α)·z_ip_i]  /  Σ w_i

        where:
          α   = blend_alpha  (population weight)
          w_i = per-feature importance weight
        """
        available = [f for f in self.FEATURE_COLS if f in features]
        if not available:
            return 0.0

        α    = self.blend_alpha
        wsum = 0.0
        score = 0.0

        for feat in available:
            val  = float(features[feat])
            w    = self.feature_weights.get(feat, 1.0)
            z_p  = self._pop_z(feat, val)
            z_ip = self._ip_z(ip, feat, val)
            score += w * (α * z_p + (1 - α) * z_ip)
            wsum  += w

        return score / wsum if wsum > 0 else 0.0

    def _adaptive_threshold(self, ip: str) -> float:
        """
        Hellinger-Distance Adaptive Threshold.

        For each feature, compute the Hellinger distance between the IP's
        current Gaussian profile and the population Gaussian. Average over
        features and scale the base threshold accordingly.

        IPs that naturally deviate from the population (e.g., power users
        with high request volumes) get a slightly higher threshold, reducing
        false positives without hiding genuine attacks.
        """
        profile = self._profiles.get(ip)
        if not profile:
            return self.anomaly_threshold

        distances = []
        for feat in self.FEATURE_COLS:
            stat = profile.get(feat)
            if stat is None or stat.w_sum < 2.0:
                continue
            if feat not in self._pop_mean:
                continue

            mu_ip, s_ip = stat.mean, max(stat.std, 1e-3)
            mu_gl, s_gl = self._pop_mean[feat], self._pop_std.get(feat, 1.0)

            s2_ip, s2_gl = s_ip ** 2, s_gl ** 2
            denom = s2_ip + s2_gl
            if denom < 1e-12:
                distances.append(0.0)
                continue

            coeff  = math.sqrt(2.0 * s_ip * s_gl / denom)
            expart = math.exp(-0.25 * (mu_ip - mu_gl) ** 2 / denom)
            h2     = max(1.0 - coeff * expart, 0.0)
            distances.append(math.sqrt(h2))

        if not distances:
            return self.anomaly_threshold

        avg_h = sum(distances) / len(distances)
        # Scale threshold up to 1.4× for highly divergent IP profiles
        scale = 1.0 + min(avg_h * 0.4, 0.4)
        return self.anomaly_threshold * scale

    # ── Public API ────────────────────────────────────────────────────────
    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Two-phase ATTP execution:

        Phase 1 — compute population baseline from full df.
        Phase 2 — stream records chronologically, score each request, update
                   the IP's time-decay-weighted behavioral profile.

        Parameters
        ----------
        df : pd.DataFrame  Must contain FEATURE_COLS + an 'ip'/'ip_address' column.

        Returns
        -------
        pd.Series  Predictions: 'attp_attack' or 'attp_benign'.
        """
        self._profiles    = {}
        self._predictions = []
        self._scores      = []

        available = [c for c in self.FEATURE_COLS if c in df.columns]
        if not available:
            raise ValueError(f"DataFrame has none of: {self.FEATURE_COLS}")

        # ── Phase 1: Population Baseline ──────────────────────────────
        print(f"    [Phase 1] Computing population baseline from {len(df):,} records...")
        self._compute_population_baseline(df, available)

        baseline_table = [
            [f, f"{self._pop_mean[f]:.4f}", f"{self._pop_std[f]:.4f}"]
            for f in available
        ]
        print(tabulate(baseline_table, headers=["Feature", "Pop Mean", "Pop Std"],
                       tablefmt="simple"))

        # ── Phase 2: Streaming Anomaly Detection ───────────────────────
        print(f"\n    [Phase 2] Streaming anomaly scoring ({len(df):,} records)...")

        ip_col = 'ip' if 'ip' in df.columns else 'ip_address'

        for idx, (_, row) in enumerate(df.iterrows()):
            ip       = str(row.get(ip_col, 'unknown'))
            features = {f: float(row.get(f, 0.0)) for f in available}

            # Score current request
            score     = self._composite_score(ip, features)
            threshold = self._adaptive_threshold(ip)
            label     = 'attp_attack' if score >= threshold else 'attp_benign'

            self._predictions.append(label)
            self._scores.append(round(score, 6))

            # Fade this IP's earlier observations, then give the newest event
            # full weight. This makes recent behaviour genuinely matter more.
            if ip not in self._profiles:
                self._profiles[ip] = {f: WeightedRunningStats() for f in self.FEATURE_COLS}
            else:
                factor = math.exp(-self.decay_lambda)
                for stat in self._profiles[ip].values():
                    stat.decay(factor)
            for feat in available:
                self._profiles[ip][feat].update(features[feat], 1.0)

        return pd.Series(self._predictions, index=df.index, name='attp_label')

    def score_series(self) -> pd.Series:
        """Return the raw composite anomaly scores from the last fit_predict call."""
        return pd.Series(self._scores, name='attp_score')


# ════════════════════════════════════════════════════════════════════════
#  Inline Feature Engineering
#  (when engineered_features.csv is absent; derives features from raw logs)
# ════════════════════════════════════════════════════════════════════════

def _engineer_features_inline(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Compute the 5 ATTP feature columns from raw labeled log columns."""
    print("[*] Computing features inline from raw log columns...")

    url_col = next((c for c in ['url_path', 'request_resource', 'request_data']
                    if c in df.columns), None)
    ua_col  = 'user_agent'   if 'user_agent'   in df.columns else None
    sc_col  = 'status_code'  if 'status_code'  in df.columns else None

    # Feature 1: Total requests per IP in this dataset
    df['ip_request_count'] = df['ip'].map(df['ip'].value_counts()).fillna(1).astype(float)
    print("    ✔ ip_request_count")

    # Feature 2: Time gap between consecutive requests per IP (seconds)
    if ts_col:
        df['time_diff_seconds'] = (
            df.groupby('ip')[ts_col].diff().dt.total_seconds().fillna(0.0)
        )
    else:
        df['time_diff_seconds'] = 0.0
    print("    ✔ time_diff_seconds")

    # Feature 3: Accumulated 404 errors per IP
    if sc_col:
        df['_is_404'] = (df[sc_col].astype(str) == '404').astype(int)
        df['status_code_404_count'] = df.groupby('ip')['_is_404'].transform('sum').astype(float)
        df.drop(columns=['_is_404'], inplace=True)
    else:
        df['status_code_404_count'] = 0.0
    print("    ✔ status_code_404_count")

    # Feature 4: Bot / scanner user-agent flag
    bot_kw = ['bot', 'crawler', 'spider', 'python', 'curl', 'wget',
              'nikto', 'sqlmap', 'nmap', 'hydra', 'gobuster']
    if ua_col:
        df['is_bot_agent'] = (
            df[ua_col].astype(str).str.lower()
                      .apply(lambda ua: 1 if any(b in ua for b in bot_kw) else 0)
                      .astype(float)
        )
    else:
        df['is_bot_agent'] = 0.0
    print("    ✔ is_bot_agent")

    # Feature 5: Shannon entropy of URL / resource path
    def _entropy(s: str) -> float:
        s = str(s)
        if not s or len(s) < 2:
            return 0.0
        prob = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * math.log2(p) for p in prob)

    if url_col:
        df['url_entropy'] = df[url_col].astype(str).apply(_entropy)
    else:
        df['url_entropy'] = 0.0
    print("    ✔ url_entropy")

    return df


# ════════════════════════════════════════════════════════════════════════
#  Evaluation Helpers
# ════════════════════════════════════════════════════════════════════════

def _binarise_ground_truth(y: pd.Series) -> pd.Series:
    """Multi-class labels → binary: benign=0, attack=1."""
    return y.apply(lambda l: 0 if str(l).lower() == 'benign' else 1)


def _binarise_attp(y: pd.Series) -> pd.Series:
    """ATTP labels → binary: attp_benign=0, attp_attack=1."""
    return y.apply(lambda l: 0 if l == 'attp_benign' else 1)


# ════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ════════════════════════════════════════════════════════════════════════

def run_attp(input_file: str = "engineered_features.csv"):
    print("=" * 68)
    print("UNIQUE ALGORITHM — Adaptive Temporal Threat Profiler (ATTP)")
    print("=" * 68)

    # ── 1. Locate and load dataset ───────────────────────────────────────
    candidates = [input_file, "engineered_features.csv",
                  "labeled_logs.csv", "cleaned_logs.csv"]
    resolved   = None
    for c in candidates:
        p = c if os.path.isabs(c) else os.path.join(BASE_DIR, c)
        if os.path.exists(p):
            resolved = p
            break

    if resolved is None:
        print("[ERROR] No input file found. Run Practical 5 first.")
        sys.exit(1)

    print(f"[*] Loading dataset from '{resolved}'...")
    df = pd.read_csv(resolved, low_memory=False)
    print(f"[+] Total records loaded: {len(df):,}")

    # ── 2. Normalise IP column ────────────────────────────────────────────
    if 'ip' not in df.columns and 'ip_address' in df.columns:
        df['ip'] = df['ip_address']
    elif 'ip' not in df.columns:
        df['ip'] = 'unknown'

    # ── 3. Parse and sort by timestamp ───────────────────────────────────
    ts_col = next((c for c in ['timestamp_dt', 'date_time', 'timestamp']
                   if c in df.columns), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        df = df.sort_values(ts_col).reset_index(drop=True)
        print(f"[+] Sorted chronologically by '{ts_col}'.")

    # ── 4. Stratified sample cap (ATTP is O(n) streaming) ────────────────
    SAMPLE_LIMIT = 100_000
    if len(df) > SAMPLE_LIMIT:
        strat_col = 'label' if 'label' in df.columns else None
        if strat_col:
            df = (
                df.groupby(strat_col, group_keys=False)
                  .apply(lambda g: g.sample(
                      n=max(1, int(SAMPLE_LIMIT * len(g) / len(df))),
                      random_state=42
                  ))
                  .reset_index(drop=True)
            )
        else:
            df = df.sample(n=SAMPLE_LIMIT, random_state=42).reset_index(drop=True)
        print(f"[*] Stratified sample: {len(df):,} records selected.")

    # ── 5. Feature engineering (inline if needed) ────────────────────────
    missing = [f for f in AdaptiveTemporalThreatProfiler.FEATURE_COLS
               if f not in df.columns]
    if missing:
        print(f"[*] Missing features {missing} — computing inline...")
        df = _engineer_features_inline(df, ts_col)

    # Fill any residual NaNs
    for feat in AdaptiveTemporalThreatProfiler.FEATURE_COLS:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0.0)

    # ── 6. Run ATTP ───────────────────────────────────────────────────────
    print(f"\n[*] Running ATTP on {len(df):,} log entries...")
    print("    Parameters:")
    print(f"      decay_lambda      = 0.001   (exponential time-decay coefficient)")
    print(f"      anomaly_threshold = 2.0     (base composite z-score cutoff)")
    print(f"      blend_alpha       = 0.70    (population vs IP-profile weight)")
    print(f"      fusion_method     = Dual-Layer Weighted Arithmetic Mean")
    print(f"      threshold_scaling = Hellinger Distance Adaptive Threshold")
    print()

    attp        = AdaptiveTemporalThreatProfiler(decay_lambda=0.001,
                                                  anomaly_threshold=2.0,
                                                  blend_alpha=0.70)
    attp_preds  = attp.fit_predict(df)
    attp_scores = attp.score_series()

    df['attp_label'] = attp_preds.values
    df['attp_score'] = attp_scores.values

    # ── 7. Prediction Distribution ───────────────────────────────────────
    pred_dist  = df['attp_label'].value_counts()
    print("\n--- ATTP Prediction Distribution ---")
    pred_table = [
        [lbl, f"{cnt:,}", f"{cnt/len(df)*100:.2f}%"]
        for lbl, cnt in pred_dist.items()
    ]
    print(tabulate(pred_table, headers=["ATTP Label", "Count", "Share (%)"],
                   tablefmt="fancy_grid"))

    # ── 8. Score Statistics ───────────────────────────────────────────────
    print("\n--- Anomaly Score Statistics ---")
    sc_stats = df['attp_score'].describe()
    print(tabulate([[k, f"{v:.4f}"] for k, v in sc_stats.items()],
                   headers=["Statistic", "Value"], tablefmt="fancy_grid"))

    # ── 9. Evaluate against ground truth ─────────────────────────────────
    if 'label' in df.columns:
        y_true_bin = _binarise_ground_truth(df['label'])
        y_pred_bin = _binarise_attp(df['attp_label'])

        accuracy  = accuracy_score(y_true_bin, y_pred_bin)
        precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall    = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1        = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        cm        = confusion_matrix(y_true_bin, y_pred_bin)

        print("\n--- ATTP vs Ground-Truth Labels (Binary: Benign vs Attack) ---")
        print(classification_report(y_true_bin, y_pred_bin,
                                    target_names=["BENIGN", "ATTACK"], digits=4))

        eval_table = [
            ["Accuracy",  f"{accuracy  * 100:.4f}%"],
            ["Precision", f"{precision * 100:.4f}%"],
            ["Recall",    f"{recall    * 100:.4f}%"],
            ["F1-Score",  f"{f1        * 100:.4f}%"],
        ]
        print(tabulate(eval_table, headers=["Metric", "Score"], tablefmt="fancy_grid"))

        print("\n--- Confusion Matrix (rows = Actual, cols = Predicted) ---")
        cm_table = [
            ["Actual BENIGN", f"TN = {cm[0,0]:,}", f"FP = {cm[0,1]:,}"],
            ["Actual ATTACK", f"FN = {cm[1,0]:,}", f"TP = {cm[1,1]:,}"],
        ]
        print(tabulate(cm_table, headers=["", "Pred BENIGN", "Pred ATTACK"],
                       tablefmt="fancy_grid"))

        # Score distribution by ground-truth class
        print("\n--- ATTP Score Distribution by Ground-Truth Class ---")
        score_by_class = df.groupby(df['label'].apply(
            lambda l: 'ATTACK' if str(l).lower() != 'benign' else 'BENIGN'
        ))['attp_score'].describe()[['mean', 'std', 'min', '50%', 'max']]
        print(tabulate(score_by_class.round(4), headers='keys', tablefmt='fancy_grid'))

    # ── 10. Top Riskiest IPs ──────────────────────────────────────────────
    ip_col = 'ip' if 'ip' in df.columns else 'ip_address'
    top_ips = (
        df.groupby(ip_col)['attp_score']
          .mean()
          .sort_values(ascending=False)
          .head(10)
          .reset_index()
    )
    top_ips.columns = ['IP Address', 'Avg ATTP Score']
    print("\n--- Top 10 Riskiest IPs (by Average ATTP Anomaly Score) ---")
    print(tabulate(
        [[r['IP Address'], f"{r['Avg ATTP Score']:.4f}"] for _, r in top_ips.iterrows()],
        headers=["IP Address", "Avg ATTP Score"],
        tablefmt="fancy_grid"
    ))

    # ── 11. Save ─────────────────────────────────────────────────────────
    out_path = os.path.join(BASE_DIR, "attp_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[+] ATTP predictions saved to '{out_path}'")

    print("\n" + "=" * 68)
    print("[✔] Adaptive Temporal Threat Profiler (ATTP) execution complete!")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unique Algorithm: Adaptive Temporal Threat Profiler (ATTP)"
    )
    parser.add_argument("--input", default="engineered_features.csv",
                        help="Input CSV (default: engineered_features.csv)")
    args = parser.parse_args()
    run_attp(args.input)
