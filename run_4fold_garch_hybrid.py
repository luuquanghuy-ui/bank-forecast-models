"""
4-Fold Walk-Forward: GARCH vs Hybrid (VOLATILITY prediction)

Các model khác (NP, TFT) predict PRICE, không so sánh trực tiếp được.
Chỉ so sánh GARCH vs Hybrid vì cả 2 đều predict VOLATILITY.
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
    """GARCH(1,1) walk-forward."""
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
    """Ridge walk-forward."""
    feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5"]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].abs().values
    X_test = test_df[feature_cols].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def hybrid_walkforward(train_df, test_df, train_ret, test_ret, garch_weight=0.5):
    """Hybrid GARCH+Ridge ensemble."""
    garch_pred = garch_walkforward(train_ret, test_ret)
    ridge_pred = ridge_walkforward(train_df, test_df)
    return garch_weight * garch_pred + (1 - garch_weight) * ridge_pred


def find_best_weight(train_df, val_df, train_ret, val_ret):
    """Find best weight for hybrid on validation set."""
    garch_val = garch_walkforward(train_ret, val_ret)
    ridge_val = ridge_walkforward(train_df, val_df)
    val_abs = np.abs(val_ret)

    best_w = 0.5
    best_mae = float('inf')

    for w in np.arange(0.0, 1.05, 0.05):
        hybrid_val = w * garch_val + (1 - w) * ridge_val
        mae = mean_absolute_error(val_abs, hybrid_val)
        if mae < best_mae:
            best_mae = mae
            best_w = w

    return best_w


def run_4fold_comparison():
    print("=" * 70)
    print("4-FOLD WALK-FORWARD: GARCH vs HYBRID (Volatility Prediction)")
    print("=" * 70)

    all_results = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        df = load_data(path)
        df = create_features(df)
        n = len(df)

        # 4 folds: expanding window
        fold_configs = [
            (0.50, 0.65, 0.70),  # Fold 1: train 50%, val 65%, test 70%
            (0.65, 0.80, 0.85),  # Fold 2: train 65%, val 80%, test 85%
            (0.80, 0.90, 0.95),  # Fold 3: train 80%, val 90%, test 95%
            (0.90, 0.95, 1.00),  # Fold 4: train 90%, val 95%, test 100%
        ]

        fold_results = []

        for fold_idx, (train_pct, val_pct, test_pct) in enumerate(fold_configs):
            train_end = int(n * train_pct)
            val_end = int(n * val_pct)
            test_end = int(n * test_pct)

            if test_end - val_end < 30 or train_end < 500:
                continue

            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:test_end].copy()

            train_ret = train_df["log_return"].values
            val_ret = val_df["log_return"].values
            test_ret = test_df["log_return"].values
            test_abs = np.abs(test_ret)

            if len(test_df) < 30:
                continue

            # Find best weight using validation
            best_w = find_best_weight(train_df, val_df, train_ret, val_ret)

            # Predictions
            naive_pred = np.zeros(len(test_ret))
            naive_mae = mean_absolute_error(test_abs, naive_pred)

            garch_pred = garch_walkforward(train_ret, test_ret)
            garch_mae = mean_absolute_error(test_abs, garch_pred)

            ridge_pred = ridge_walkforward(train_df, test_df)
            ridge_mae = mean_absolute_error(test_abs, ridge_pred)

            hybrid_pred = best_w * garch_pred + (1 - best_w) * ridge_pred
            hybrid_mae = mean_absolute_error(test_abs, hybrid_pred)

            fold_results.append({
                "fold": fold_idx + 1,
                "train_n": train_end,
                "val_n": val_end - train_end,
                "test_n": test_end - val_end,
                "best_w": best_w,
                "naive_mae": naive_mae,
                "garch_mae": garch_mae,
                "ridge_mae": ridge_mae,
                "hybrid_mae": hybrid_mae,
            })

            print(f"  Fold {fold_idx+1}: train={train_end}, val={val_end-train_end}, test={test_end-val_end}, best_w={best_w:.2f}")
            print(f"    Naive:  {naive_mae:.6f}")
            print(f"    GARCH:  {garch_mae:.6f}")
            print(f"    Ridge:  {ridge_mae:.6f}")
            print(f"    Hybrid: {hybrid_mae:.6f}")

        if fold_results:
            df_folds = pd.DataFrame(fold_results)
            avg = df_folds.mean(numeric_only=True)

            print(f"\n  AVERAGE:")
            print(f"    Naive:  {avg['naive_mae']:.6f}")
            print(f"    GARCH:  {avg['garch_mae']:.6f}")
            print(f"    Ridge:  {avg['ridge_mae']:.6f}")
            print(f"    Hybrid: {avg['hybrid_mae']:.6f}")

            winner = min(["GARCH", "Ridge", "Hybrid"], key=lambda x: avg[f"{x.lower()}_mae"])
            print(f"    Winner: {winner}")

            df_folds.to_csv(OUTPUT_DIR / f"{bank}_4fold_garch_hybrid.csv", index=False)

            all_results.append({
                "bank": bank,
                "n_folds": len(fold_results),
                "avg_naive": avg["naive_mae"],
                "avg_garch": avg["garch_mae"],
                "avg_ridge": avg["ridge_mae"],
                "avg_hybrid": avg["hybrid_mae"],
                "winner": winner,
            })

    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(OUTPUT_DIR / "4fold_garch_hybrid_summary.csv", index=False)

    # Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_summary))
    width = 0.2

    ax.bar(x - 1.5*width, df_summary["avg_naive"], width, label="Naive", color="gray")
    ax.bar(x - 0.5*width, df_summary["avg_garch"], width, label="GARCH", color="blue")
    ax.bar(x + 0.5*width, df_summary["avg_ridge"], width, label="Ridge", color="green")
    ax.bar(x + 1.5*width, df_summary["avg_hybrid"], width, label="Hybrid", color="red")

    ax.set_ylabel("MAE")
    ax.set_title("4-Fold Walk-Forward: Volatility Prediction")
    ax.set_xticks(x)
    ax.set_xticklabels(df_summary["bank"])
    ax.legend()

    for i, r in df_summary.iterrows():
        ax.annotate(f"W: {r['winner']}", (i, r[f"avg_{r['winner'].lower()}"] + 0.001),
                   ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4fold_garch_hybrid_comparison.png", dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for _, r in df_summary.iterrows():
        print(f"\n{r['bank']}: Winner = {r['winner']}")
        print(f"  Naive: {r['avg_naive']:.6f}")
        print(f"  GARCH: {r['avg_garch']:.6f}")
        print(f"  Ridge: {r['avg_ridge']:.6f}")
        print(f"  Hybrid: {r['avg_hybrid']:.6f}")

    print(f"\n\nSaved: {OUTPUT_DIR}/4fold_garch_hybrid_summary.csv")
    return df_summary


if __name__ == "__main__":
    run_4fold_comparison()