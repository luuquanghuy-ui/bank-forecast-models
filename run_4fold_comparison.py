"""
4-Fold Walk-Forward Comparison: All Models

Fold 1: Train 1, Val 2, Test 3 (skip 4)
Fold 2: Train 1-2, Val 3, Test 4
Fold 3: Train 1-3, Val 4, Test 5 (if exists)
Fold 4: Train 1-4, Val 5, Test 6 (if exists)

Models:
1. Naive (baseline)
2. GARCH(1,1) - volatility
3. Ridge Regression - ML baseline
4. Hybrid GARCH+Ridge - best model
5. NeuralProphet - deep learning (price prediction)
6. TFT - attention (price prediction)
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from arch import arch_model
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = BASE_DIR / "four_fold_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"date": "ds", "close": "y"})
    df["log_return"] = np.log(df["y"] / df["y"].shift(1))
    return df.dropna().reset_index(drop=True)


def create_features(df):
    """Create features for ML models."""
    data = df.copy()
    data["volume_lag1"] = data["volume"].shift(1)
    data["volatility_5d"] = data["log_return"].rolling(5).std().shift(1)
    data["volatility_20d"] = data["log_return"].rolling(20).std().shift(1)
    data["rsi_lag1"] = data["rsi"].shift(1)
    data["return_lag1"] = data["log_return"].shift(1)
    data["return_lag2"] = data["log_return"].shift(2)
    data["return_lag5"] = data["log_return"].shift(5)
    return data.dropna().reset_index(drop=True)


def garch_walkforward(train_ret, test_ret):
    """GARCH(1,1) walk-forward prediction."""
    ret_scaled = train_ret * 100.0
    model = arch_model(ret_scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = model.fit(disp="off", show_warning=False)

    mu = float(res.params.get("mu", res.params.get("Const", 0.0))) / 100.0
    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])

    sigma2 = np.empty(len(test_ret))
    sigma2_last = max(float(np.var(ret_scaled)), 1e-6)
    eps_last = (test_ret[0] - mu) * 100.0

    for i in range(len(test_ret)):
        sigma2[i] = omega + alpha * (eps_last ** 2) + beta * sigma2_last
        eps_last = (test_ret[i] - mu) * 100.0
        sigma2_last = sigma2[i]

    return np.sqrt(sigma2) / 100.0


def ridge_walkforward(train_df, test_df, target_col="log_return"):
    """Ridge walk-forward prediction."""
    feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5"]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].abs().values  # Predict absolute return (volatility proxy)
    X_test = test_df[feature_cols].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return pred


def hybrid_walkforward(train_df, test_df, train_ret, test_ret, garch_weight=0.5):
    """Hybrid GARCH+Ridge ensemble."""
    garch_pred = garch_walkforward(train_ret, test_ret)
    ridge_pred = ridge_walkforward(train_df, test_df)

    # Ensemble: weighted average
    hybrid_pred = garch_weight * garch_pred + (1 - garch_weight) * ridge_pred
    return hybrid_pred


def run_4fold_comparison():
    """Run 4-fold walk-forward comparison."""
    print("=" * 70)
    print("4-FOLD WALK-FORWARD COMPARISON")
    print("=" * 70)

    all_results = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        df = load_data(path)
        df = create_features(df)
        n = len(df)

        # 4-fold walk-forward
        # Fold 1: indices 0-50%, test 50-65%
        # Fold 2: indices 0-65%, test 65-80%
        # Fold 3: indices 0-80%, test 80-100% (or last portion)
        fold_configs = [
            (0.50, 0.65),  # Fold 1: 50% train, test 50-65%
            (0.65, 0.80),  # Fold 2: 65% train, test 65-80%
            (0.80, 0.90),  # Fold 3: 80% train, test 80-90%
            (0.90, 1.00),  # Fold 4: 90% train, test 90-100%
        ]

        fold_results = []

        for fold_idx, (train_end_pct, test_end_pct) in enumerate(fold_configs):
            train_end = int(n * train_end_pct)
            test_end = int(n * test_end_pct)

            # Ensure minimum sizes
            if train_end < int(n * 0.4) or test_end - train_end < 50:
                continue

            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            train_ret = train_df["log_return"].values
            test_ret = test_df["log_return"].values
            test_abs = np.abs(test_ret)

            if len(test_df) < 50:
                continue

            # Predictions
            naive_pred = np.zeros(len(test_ret))

            try:
                garch_pred = garch_walkforward(train_ret, test_ret)
                garch_mae = mean_absolute_error(test_abs, garch_pred)
            except:
                garch_pred = np.zeros(len(test_ret))
                garch_mae = np.nan

            try:
                ridge_pred = ridge_walkforward(train_df, test_df)
                ridge_mae = mean_absolute_error(test_abs, ridge_pred)
            except:
                ridge_pred = np.zeros(len(test_ret))
                ridge_mae = np.nan

            try:
                hybrid_pred = hybrid_walkforward(train_df, test_df, train_ret, test_ret, garch_weight=0.5)
                hybrid_mae = mean_absolute_error(test_abs, hybrid_pred)
            except:
                hybrid_pred = np.zeros(len(test_ret))
                hybrid_mae = np.nan

            naive_mae = mean_absolute_error(test_abs, naive_pred)

            fold_results.append({
                "fold": fold_idx + 1,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "n_train": train_end,
                "n_test": test_end - train_end,
                "naive_mae": naive_mae,
                "garch_mae": garch_mae,
                "ridge_mae": ridge_mae,
                "hybrid_mae": hybrid_mae,
            })

            print(f"  Fold {fold_idx+1}: train={train_end}, test=({train_end},{test_end}), n_test={test_end-train_end}")
            print(f"    Naive MAE: {naive_mae:.6f}")
            print(f"    GARCH MAE: {garch_mae:.6f}")
            print(f"    Ridge MAE: {ridge_mae:.6f}")
            print(f"    Hybrid MAE: {hybrid_mae:.6f}")

        # Summary
        if fold_results:
            df_folds = pd.DataFrame(fold_results)
            avg_results = df_folds.mean(numeric_only=True)

            print(f"\n  AVERAGE across {len(fold_results)} folds:")
            print(f"    Naive:  {avg_results['naive_mae']:.6f}")
            print(f"    GARCH:  {avg_results['garch_mae']:.6f}")
            print(f"    Ridge:  {avg_results['ridge_mae']:.6f}")
            print(f"    Hybrid: {avg_results['hybrid_mae']:.6f}")

            # Save fold results
            df_folds.to_csv(OUTPUT_DIR / f"{bank}_4fold_results.csv", index=False)

            # Store summary
            all_results.append({
                "bank": bank,
                "n_folds": len(fold_results),
                "naive_avg": avg_results["naive_mae"],
                "garch_avg": avg_results["garch_mae"],
                "ridge_avg": avg_results["ridge_mae"],
                "hybrid_avg": avg_results["hybrid_mae"],
                "garch_vs_naive": (avg_results["naive_mae"] - avg_results["garch_mae"]) / avg_results["naive_mae"] * 100,
                "hybrid_vs_naive": (avg_results["naive_mae"] - avg_results["hybrid_mae"]) / avg_results["naive_mae"] * 100,
            })

    # Save summary
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(OUTPUT_DIR / "4fold_summary.csv", index=False)

    # Create comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    banks = df_summary["bank"].values
    x = np.arange(len(banks))
    width = 0.2

    ax = axes[0]
    ax.bar(x - 1.5*width, df_summary["naive_avg"], width, label="Naive", color="gray")
    ax.bar(x - 0.5*width, df_summary["garch_avg"], width, label="GARCH", color="blue")
    ax.bar(x + 0.5*width, df_summary["ridge_avg"], width, label="Ridge", color="green")
    ax.bar(x + 1.5*width, df_summary["hybrid_avg"], width, label="Hybrid", color="red")
    ax.set_xlabel("Bank")
    ax.set_ylabel("MAE")
    ax.set_title("Average MAE across 4 Folds")
    ax.set_xticks(x)
    ax.set_xticklabels(banks)
    ax.legend()

    ax2 = axes[1]
    ax2.bar(x, df_summary["garch_vs_naive"], width, label="GARCH vs Naive", color="blue")
    ax2.bar(x, df_summary["hybrid_vs_naive"], width, label="Hybrid vs Naive", color="red")
    ax2.set_xlabel("Bank")
    ax2.set_ylabel("Improvement (%)")
    ax2.set_title("MAE Reduction vs Naive (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(banks)
    ax2.legend()

    # Table of results
    ax3 = axes[2]
    ax3.axis("off")
    table_data = []
    for _, r in df_summary.iterrows():
        table_data.append([
            r["bank"],
            f"{r['naive_avg']:.4f}",
            f"{r['garch_avg']:.4f}",
            f"{r['ridge_avg']:.4f}",
            f"{r['hybrid_avg']:.4f}",
        ])
    table = ax3.table(cellText=table_data, colLabels=["Bank", "Naive", "GARCH", "Ridge", "Hybrid"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax3.set_title("MAE Summary")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4fold_comparison.png", dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("4-FOLD SUMMARY")
    print("=" * 70)
    print(f"\nSaved to: {OUTPUT_DIR}/")
    for _, r in df_summary.iterrows():
        print(f"\n{r['bank']}:")
        print(f"  Naive:  {r['naive_avg']:.6f}")
        print(f"  GARCH:  {r['garch_avg']:.6f} ({(r['garch_vs_naive']):.1f}% better than Naive)")
        print(f"  Ridge:  {r['ridge_avg']:.6f}")
        print(f"  Hybrid: {r['hybrid_avg']:.6f} ({(r['hybrid_vs_naive']):.1f}% better than Naive)")

    return df_summary


if __name__ == "__main__":
    results = run_4fold_comparison()