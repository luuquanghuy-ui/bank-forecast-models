"""
Lăng kính 1: XGBoost + SHAP — "Cái gì chi phối mức độ biến động ngân hàng VN?"

Target: |log_return| (absolute daily return, proxy cho daily volatility)
Method: XGBoost + SHAP → feature importance + cross-bank DNA comparison

Chạy: python langkinh1_xgboost_shap.py
Output: langkinh1_xgboost_shap/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    print(f"XGBoost version: {xgb.__version__}")
except ImportError:
    print("ERROR: pip install xgboost")
    raise SystemExit(1)

try:
    import shap
    print(f"SHAP version: {shap.__version__}")
except ImportError:
    print("ERROR: pip install shap")
    raise SystemExit(1)

from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "langkinh1_xgboost_shap"
OUTPUT_DIR.mkdir(exist_ok=True)

BANKS = ['BID', 'CTG', 'VCB']
BANK_FILES = {t: SCRIPT_DIR / f"banks_{t}_dataset.csv" for t in BANKS}

# Feature groups
TECHNICAL_FEATURES = [
    'return_lag1', 'return_lag2', 'return_lag3', 'return_lag5',
    'volatility_lag1', 'volatility_lag2',
    'rsi_lag1',
    'volume_lag1', 'volume_ratio',
    'ma_ratio', 'ma50_ratio',
]

MACRO_FEATURES = [
    'vnindex_lag1', 'vn30_lag1',
    'usd_vnd_lag1', 'interest_rate_lag1',
]

ALL_FEATURES = TECHNICAL_FEATURES + MACRO_FEATURES

# Readable names for plots
FEATURE_LABELS = {
    'return_lag1': 'Return (t-1)',
    'return_lag2': 'Return (t-2)',
    'return_lag3': 'Return (t-3)',
    'return_lag5': 'Return (t-5)',
    'volatility_lag1': 'Volatility 20d (t-1)',
    'volatility_lag2': 'Volatility 20d (t-2)',
    'rsi_lag1': 'RSI (t-1)',
    'volume_lag1': 'Volume (t-1)',
    'volume_ratio': 'Volume Ratio',
    'ma_ratio': 'Price / MA20',
    'ma50_ratio': 'Price / MA50',
    'vnindex_lag1': 'VNIndex (t-1)',
    'vn30_lag1': 'VN30 (t-1)',
    'usd_vnd_lag1': 'USD/VND (t-1)',
    'interest_rate_lag1': 'Interest Rate (t-1)',
}

COLORS = {
    'BID': '#2E86AB',
    'CTG': '#A23B72',
    'VCB': '#F18F01',
    'Technical': '#2ecc71',
    'Macro': '#3498db',
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
def load_and_prepare(ticker):
    """Load bank dataset and create features."""
    path = BANK_FILES[ticker]
    df = pd.read_csv(path, parse_dates=['date'], encoding='utf-8').sort_values('date').reset_index(drop=True)

    # Target: absolute return (proxy for daily volatility)
    df['target'] = np.abs(df['log_return'])

    # Lagged returns
    for lag in [1, 2, 3, 5]:
        df[f'return_lag{lag}'] = df['log_return'].shift(lag)

    # Volatility lags
    df['volatility_lag1'] = df['volatility_20d'].shift(1)
    df['volatility_lag2'] = df['volatility_20d'].shift(2)

    # RSI lag
    df['rsi_lag1'] = df['rsi'].shift(1)

    # Volume features
    df['volume_lag1'] = df['volume'].shift(1)
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # MA ratio features
    df['ma_ratio'] = df['close'] / df['ma20']
    df['ma50_ratio'] = df['close'] / df['ma50']

    # Macro lags
    df['vnindex_lag1'] = df['vnindex_close'].shift(1)
    df['vn30_lag1'] = df['vn30_close'].shift(1)
    df['usd_vnd_lag1'] = df['usd_vnd'].shift(1)
    df['interest_rate_lag1'] = df['interest_rate'].shift(1)

    df = df.dropna().reset_index(drop=True)
    return df


# ============================================================
# 2. TRAIN XGBOOST + SHAP
# ============================================================
def train_and_shap(df, ticker):
    """Train XGBoost on target=|log_return|, compute SHAP on test set."""
    X = df[ALL_FEATURES].values
    y = df['target'].values
    n = len(X)

    # Walk-forward split: 70/15/15
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  Test MAE: {mae:.6f}, R²: {r2:.4f}")

    # SHAP
    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return model, shap_values, X_test, y_test, mae, r2


# ============================================================
# 3. PART A — Technical vs Macro
# ============================================================
def analyze_groups(shap_values, ticker):
    """Calculate group-level SHAP importance."""
    mean_shap = np.abs(shap_values).mean(axis=0)

    tech_idx = [ALL_FEATURES.index(f) for f in TECHNICAL_FEATURES]
    macro_idx = [ALL_FEATURES.index(f) for f in MACRO_FEATURES]

    tech_total = mean_shap[tech_idx].sum()
    macro_total = mean_shap[macro_idx].sum()
    total = tech_total + macro_total

    tech_pct = tech_total / total * 100
    macro_pct = macro_total / total * 100

    print(f"  Technical: {tech_pct:.1f}% | Macro: {macro_pct:.1f}%")

    return {
        'ticker': ticker,
        'tech_total': tech_total,
        'macro_total': macro_total,
        'tech_pct': tech_pct,
        'macro_pct': macro_pct,
    }


def analyze_features(shap_values, ticker):
    """Calculate per-feature SHAP importance and ranking."""
    mean_shap = np.abs(shap_values).mean(axis=0)
    std_shap = np.abs(shap_values).std(axis=0)

    results = pd.DataFrame({
        'feature': ALL_FEATURES,
        'label': [FEATURE_LABELS[f] for f in ALL_FEATURES],
        'group': ['Technical' if f in TECHNICAL_FEATURES else 'Macro' for f in ALL_FEATURES],
        'mean_abs_shap': mean_shap,
        'std_abs_shap': std_shap,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    results['rank'] = range(1, len(results) + 1)

    return results


# ============================================================
# 4. PLOTS
# ============================================================
def plot_group_comparison(all_groups):
    """Pie + bar chart: Technical vs Macro per bank."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for i, g in enumerate(all_groups):
        ticker = g['ticker']
        sizes = [g['tech_pct'], g['macro_pct']]
        labels = [f"Technical\n{g['tech_pct']:.1f}%", f"Macro\n{g['macro_pct']:.1f}%"]
        colors = [COLORS['Technical'], COLORS['Macro']]

        axes[i].pie(sizes, labels=labels, colors=colors, startangle=90,
                    textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[i].set_title(f'{ticker}', fontsize=14, fontweight='bold')

    plt.suptitle('Part A: Technical vs Macro — Feature Group Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'partA_group_comparison.png')
    plt.close()
    print(f"  Saved: partA_group_comparison.png")


def plot_feature_bars(all_feature_results):
    """Horizontal bar chart per bank showing all features ranked by SHAP."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for i, (ticker, df) in enumerate(all_feature_results.items()):
        df_plot = df.sort_values('mean_abs_shap', ascending=True)
        colors = [COLORS['Technical'] if g == 'Technical' else COLORS['Macro'] for g in df_plot['group']]

        axes[i].barh(df_plot['label'], df_plot['mean_abs_shap'], color=colors, edgecolor='white')
        axes[i].set_xlabel('Mean |SHAP|')
        axes[i].set_title(f'{ticker}', fontsize=13, fontweight='bold')

        # Value labels
        for idx, (val, label) in enumerate(zip(df_plot['mean_abs_shap'], df_plot['label'])):
            axes[i].text(val + df_plot['mean_abs_shap'].max() * 0.02, idx,
                        f'{val:.5f}', va='center', fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['Technical'], label='Technical'),
                       Patch(facecolor=COLORS['Macro'], label='Macro')]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.suptitle('Part A: Individual Feature Importance (Mean |SHAP|)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'partA_feature_bars.png')
    plt.close()
    print(f"  Saved: partA_feature_bars.png")


def plot_cross_bank_heatmap(all_feature_results):
    """Part B: Heatmap of SHAP rankings across banks."""
    # Build rank table
    rank_data = {}
    for ticker, df in all_feature_results.items():
        rank_data[ticker] = {row['label']: row['rank'] for _, row in df.iterrows()}

    labels = [FEATURE_LABELS[f] for f in ALL_FEATURES]
    rank_matrix = np.array([[rank_data[t][label] for t in BANKS] for label in labels])

    fig, ax = plt.subplots(figsize=(8, 9))

    # Heatmap: lower rank (more important) = darker color
    im = ax.imshow(rank_matrix, cmap='YlOrRd_r', aspect='auto', vmin=1, vmax=15)

    ax.set_xticks(range(len(BANKS)))
    ax.set_xticklabels(BANKS, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate with rank numbers
    for i in range(len(labels)):
        for j in range(len(BANKS)):
            rank = rank_matrix[i, j]
            color = 'white' if rank <= 5 else 'black'
            fontweight = 'bold' if rank <= 3 else 'normal'
            ax.text(j, i, f'#{rank}', ha='center', va='center',
                    color=color, fontsize=10, fontweight=fontweight)

    plt.colorbar(im, label='Rank (1 = most important)', shrink=0.8)
    ax.set_title('Part B: Cross-Bank DNA — SHAP Ranking Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'partB_crossbank_heatmap.png')
    plt.close()
    print(f"  Saved: partB_crossbank_heatmap.png")


def plot_cross_bank_table(all_feature_results):
    """Part B: Side-by-side ranking table highlighting differences."""
    # Build comparison dataframe
    rows = []
    for f in ALL_FEATURES:
        label = FEATURE_LABELS[f]
        group = 'Tech' if f in TECHNICAL_FEATURES else 'Macro'
        row = {'Feature': label, 'Group': group}
        for ticker, df in all_feature_results.items():
            feat_row = df[df['feature'] == f].iloc[0]
            row[f'{ticker}_rank'] = int(feat_row['rank'])
            row[f'{ticker}_shap'] = feat_row['mean_abs_shap']
        # Rank variance (how different across banks)
        ranks = [row[f'{t}_rank'] for t in BANKS]
        row['rank_variance'] = np.var(ranks)
        rows.append(row)

    comp_df = pd.DataFrame(rows).sort_values('rank_variance', ascending=False)
    comp_df.to_csv(OUTPUT_DIR / 'partB_crossbank_comparison.csv', index=False)
    print(f"  Saved: partB_crossbank_comparison.csv")

    # Print most different features
    print("\n  Features with BIGGEST rank differences across banks:")
    for _, row in comp_df.head(5).iterrows():
        ranks_str = ", ".join([f"{t}=#{row[f'{t}_rank']}" for t in BANKS])
        print(f"    {row['Feature']}: {ranks_str} (variance={row['rank_variance']:.1f})")

    return comp_df


def plot_shap_summary_per_bank(shap_values, X_test, ticker):
    """SHAP summary plot (beeswarm) for each bank."""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, feature_names=[FEATURE_LABELS[f] for f in ALL_FEATURES],
                      show=False, max_display=15)
    plt.title(f'{ticker} — SHAP Summary (target = |log_return|)', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{ticker}_shap_summary.png')
    plt.close()
    print(f"  Saved: {ticker}_shap_summary.png")


def plot_top5_dependence(shap_values, X_test, ticker, feature_results):
    """Part C: SHAP dependence plots for top 5 features."""
    top5 = feature_results.head(5)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for i, (_, row) in enumerate(top5.iterrows()):
        feat_idx = ALL_FEATURES.index(row['feature'])
        axes[i].scatter(X_test[:, feat_idx], shap_values[:, feat_idx],
                       alpha=0.3, s=8, c='#2E86AB')
        axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.4)
        axes[i].set_xlabel(row['label'], fontsize=9)
        axes[i].set_ylabel('SHAP value' if i == 0 else '')
        axes[i].set_title(f"#{row['rank']}: {row['label']}", fontsize=10)

    plt.suptitle(f'{ticker} — Part C: Top 5 Feature Dependence', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{ticker}_partC_top5_dependence.png')
    plt.close()
    print(f"  Saved: {ticker}_partC_top5_dependence.png")


def plot_radar_dna(all_feature_results):
    """Radar chart comparing normalized SHAP profiles across banks."""
    # Use top 8 features (by average rank across banks)
    avg_ranks = {}
    for f in ALL_FEATURES:
        label = FEATURE_LABELS[f]
        ranks = []
        for ticker, df in all_feature_results.items():
            feat_row = df[df['feature'] == f].iloc[0]
            ranks.append(feat_row['rank'])
        avg_ranks[label] = np.mean(ranks)

    top8_labels = sorted(avg_ranks, key=avg_ranks.get)[:8]

    # Get normalized SHAP values for radar
    angles = np.linspace(0, 2 * np.pi, len(top8_labels), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for ticker, df in all_feature_results.items():
        values = []
        for label in top8_labels:
            row = df[df['label'] == label].iloc[0]
            values.append(row['mean_abs_shap'])
        # Normalize to [0, 1] relative to max across all banks for this feature
        max_val = max(values) if max(values) > 0 else 1
        values_norm = [v / max_val for v in values]
        values_norm += values_norm[:1]  # close

        ax.plot(angles, values_norm, 'o-', linewidth=2, label=ticker, color=COLORS[ticker])
        ax.fill(angles, values_norm, alpha=0.1, color=COLORS[ticker])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top8_labels, fontsize=9)
    ax.set_title('Cross-Bank DNA Radar (Top 8 Features, Normalized)', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'partB_radar_dna.png')
    plt.close()
    print(f"  Saved: partB_radar_dna.png")


# ============================================================
# 5. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("LĂNG KÍNH 1: XGBoost + SHAP")
    print("Câu hỏi: Cái gì chi phối mức độ biến động ngân hàng VN?")
    print("Target: |log_return| (absolute daily return)")
    print("=" * 60)

    all_shap = {}
    all_X_test = {}
    all_groups = []
    all_feature_results = {}
    all_metrics = {}

    for ticker in BANKS:
        print(f"\n{'#'*50}")
        print(f"# {ticker}")
        print(f"{'#'*50}")

        # Load
        df = load_and_prepare(ticker)
        print(f"  Data: {len(df)} rows, date range: {df['date'].min().date()} → {df['date'].max().date()}")

        # Train + SHAP
        model, shap_values, X_test, y_test, mae, r2 = train_and_shap(df, ticker)
        all_shap[ticker] = shap_values
        all_X_test[ticker] = X_test
        all_metrics[ticker] = {'mae': mae, 'r2': r2}

        # Part A: Group analysis
        group_result = analyze_groups(shap_values, ticker)
        all_groups.append(group_result)

        # Per-feature analysis
        feature_results = analyze_features(shap_values, ticker)
        all_feature_results[ticker] = feature_results

        # Save per-bank CSV
        feature_results.to_csv(OUTPUT_DIR / f'{ticker}_feature_importance.csv', index=False)

        # Save group CSV
        pd.DataFrame([group_result]).to_csv(OUTPUT_DIR / f'{ticker}_group_importance.csv', index=False)

        # SHAP summary plot
        plot_shap_summary_per_bank(shap_values, X_test, ticker)

        # Part C: Top 5 dependence
        plot_top5_dependence(shap_values, X_test, ticker, feature_results)

    # ===== CROSS-BANK ANALYSIS =====
    print(f"\n{'='*60}")
    print("CROSS-BANK ANALYSIS")
    print(f"{'='*60}")

    # Part A: Group comparison chart
    plot_group_comparison(all_groups)
    plot_feature_bars(all_feature_results)

    # Part B: Cross-bank DNA
    plot_cross_bank_heatmap(all_feature_results)
    comp_df = plot_cross_bank_table(all_feature_results)
    plot_radar_dna(all_feature_results)

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*60}")
    print("KẾT QUẢ CHÍNH")
    print(f"{'='*60}")

    print("\n1. Model Performance:")
    for ticker in BANKS:
        m = all_metrics[ticker]
        print(f"   {ticker}: MAE={m['mae']:.6f}, R²={m['r2']:.4f}")

    print("\n2. Technical vs Macro:")
    for g in all_groups:
        print(f"   {g['ticker']}: Technical {g['tech_pct']:.1f}% | Macro {g['macro_pct']:.1f}%")

    print("\n3. Top 3 Features per Bank:")
    for ticker in BANKS:
        top3 = all_feature_results[ticker].head(3)
        features_str = ", ".join([f"#{r['rank']} {r['label']}" for _, r in top3.iterrows()])
        print(f"   {ticker}: {features_str}")

    print("\n4. Biggest Cross-Bank Differences:")
    for _, row in comp_df.head(3).iterrows():
        ranks_str = " | ".join([f"{t}=#{row[f'{t}_rank']}" for t in BANKS])
        print(f"   {row['Feature']}: {ranks_str}")

    print(f"\n\nAll output saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
