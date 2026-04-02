"""
Random Forest Model for Volatility Prediction

Compare RF vs Ridge (linear vs non-linear) for predicting |log_return|.
Also create GARCH+RF ensemble.
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from arch import arch_model
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "rf_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}

RESULTS_CSV = SCRIPT_DIR / "rf_results.csv"
REPORT_MD = SCRIPT_DIR / "rf_report.md"


def load_data(path: Path) -> pd.DataFrame:
    """Load data như script gốc."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"date": "ds", "close": "y"})
    df["log_return"] = np.log(df["y"] / df["y"].shift(1))
    return df.dropna().reset_index(drop=True)


def split_data(df):
    n = len(df)
    return df.iloc[:int(n*0.70)].copy(), df.iloc[int(n*0.70):int(n*0.85)].copy(), df.iloc[int(n*0.85):].copy()


def garch_walkforward(train_ret, test_ret):
    """GARCH(1,1) walk-forward: fit on train, predict test iteratively."""
    ret_scaled = train_ret * 100.0
    model = arch_model(ret_scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = model.fit(disp="off", show_warning=False)

    mu = float(res.params.get("mu", res.params.get("Const", 0.0))) / 100.0
    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])

    # Walk-forward prediction
    sigma2 = np.empty(len(test_ret))
    sigma2_last = max(float(np.var(ret_scaled)), 1e-8)
    eps_last = (test_ret[0] - mu) * 100.0

    for i in range(len(test_ret)):
        sigma2[i] = omega + alpha * (eps_last ** 2) + beta * sigma2_last
        eps_last = (test_ret[i] - mu) * 100.0
        sigma2_last = sigma2[i]

    return np.sqrt(sigma2) / 100.0


def main():
    all_rows = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        df = load_data(path)
        train, val, test = split_data(df)

        train_ret = train["log_return"].values
        val_ret = val["log_return"].values
        test_ret = test["log_return"].values
        test_price = test["y"].values
        val_price = val["y"].values

        # Naive baseline: predict last known return = 0
        val_naive_pred = np.zeros(len(val_ret))
        test_naive_pred = np.zeros(len(test_ret))

        # GARCH walk-forward
        val_garch_sigma = garch_walkforward(train_ret, val_ret)
        test_garch_sigma = garch_walkforward(
            np.concatenate([train_ret, val_ret]),
            test_ret
        )
        val_garch_pred = np.sqrt(2 / np.pi) * val_garch_sigma
        test_garch_pred = np.sqrt(2 / np.pi) * test_garch_sigma

        # Features (same as Ridge in run_ensemble.py)
        features = ["volume", "rsi", "volatility_20d", "vnindex_close", "vn30_close", "usd_vnd", "interest_rate"]
        available = [f for f in features if f in df.columns]

        def make_features(data):
            X = data[available].values
            for i in range(1, 6):
                X = np.column_stack([X, data["log_return"].shift(i).fillna(0).values])
            return X

        X_train = make_features(train)
        X_val = make_features(val)
        X_test = make_features(test)

        # Random Forest (non-linear)
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, train_ret)
        val_rf_pred = rf.predict(X_val)
        test_rf_pred = rf.predict(X_test)

        # Metrics on |return| (volatility proxy)
        val_abs_actual = np.abs(val_ret)
        test_abs_actual = np.abs(test_ret)

        metrics = {
            "Naive": {
                "val_mae": mean_absolute_error(val_abs_actual, np.zeros(len(val_ret))),
                "test_mae": mean_absolute_error(test_abs_actual, np.zeros(len(test_ret))),
            },
            "RF": {
                "val_mae": mean_absolute_error(val_abs_actual, np.abs(val_rf_pred)),
                "test_mae": mean_absolute_error(test_abs_actual, np.abs(test_rf_pred)),
            },
            "GARCH": {
                "val_mae": mean_absolute_error(val_abs_actual, val_garch_pred),
                "test_mae": mean_absolute_error(test_abs_actual, test_garch_pred),
            },
        }

        # Ensemble GARCH+RF: weighted average
        best_w, best_mae = 0, float('inf')
        for w in np.arange(0, 1.05, 0.05):
            pred = w * val_garch_pred + (1-w) * np.abs(val_rf_pred)
            mae = mean_absolute_error(val_abs_actual, pred)
            if mae < best_mae:
                best_mae = mae
                best_w = w

        test_ensemble_pred = best_w * test_garch_pred + (1-best_w) * np.abs(test_rf_pred)
        metrics["GARCH+RF"] = {
            "val_mae": mean_absolute_error(val_abs_actual, best_w * val_garch_pred + (1-best_w) * np.abs(val_rf_pred)),
            "test_mae": mean_absolute_error(test_abs_actual, test_ensemble_pred),
            "best_w": best_w,
        }

        print(f"  Naive:     val_MAE={metrics['Naive']['val_mae']:.6f}, test_MAE={metrics['Naive']['test_mae']:.6f}")
        print(f"  RF:        val_MAE={metrics['RF']['val_mae']:.6f}, test_MAE={metrics['RF']['test_mae']:.6f}")
        print(f"  GARCH:     val_MAE={metrics['GARCH']['val_mae']:.6f}, test_MAE={metrics['GARCH']['test_mae']:.6f}")
        print(f"  GARCH+RF:  val_MAE={metrics['GARCH+RF']['val_mae']:.6f}, test_MAE={metrics['GARCH+RF']['test_mae']:.6f} (w={best_w:.2f})")

        row = {
            "bank": bank,
            "train": len(train), "val": len(val), "test": len(test),
            "naive_test_mae": metrics["Naive"]["test_mae"],
            "rf_test_mae": metrics["RF"]["test_mae"],
            "garch_test_mae": metrics["GARCH"]["test_mae"],
            "ensemble_test_mae": metrics["GARCH+RF"]["test_mae"],
            "ensemble_weight": best_w,
        }
        all_rows.append(row)

        # Feature importance
        feat_imp = pd.DataFrame({
            "feature": available + [f"lag_{i}" for i in range(1, 6)],
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
        print(f"\n  Top features for {bank}:")
        for _, r in feat_imp.head(5).iterrows():
            print(f"    {r['feature']}: {r['importance']:.4f}")

        # Plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(test["ds"].values, test_abs_actual, label="Actual |return|", alpha=0.7)
        plt.plot(test["ds"].values, test_garch_pred, label="GARCH", alpha=0.7)
        plt.plot(test["ds"].values, np.abs(test_rf_pred), label="RF", alpha=0.7)
        plt.plot(test["ds"].values, test_ensemble_pred, label="GARCH+RF Ensemble", linewidth=2)
        plt.title(f"{bank} - Test Predictions")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.barh(feat_imp["feature"].head(10), feat_imp["importance"].head(10))
        plt.title(f"{bank} - Feature Importance (RF)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{bank}_rf_test.png", dpi=150)
        plt.close()

    # Save
    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")

    lines = ["# Random Forest Model - Results\n"]
    lines.append("## Approach: Random Forest for Volatility Prediction\n")
    lines.append("Target: |log_return| (volatility proxy)\n")
    lines.append("\n| Bank | Naive MAE | RF MAE | GARCH MAE | Ensemble MAE | Ensemble Weight (GARCH) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in all_rows:
        lines.append(f"| {r['bank']} | {r['naive_test_mae']:.6f} | {r['rf_test_mae']:.6f} | "
                     f"{r['garch_test_mae']:.6f} | {r['ensemble_test_mae']:.6f} | {r['ensemble_weight']:.2f} |")

    lines.append("\n## Interpretation")
    lines.append("- RF: Random Forest captures non-linear patterns")
    lines.append("- GARCH: best for volatility (volatility clustering is real)")
    lines.append("- Ensemble: combines GARCH + RF via weight selected on validation")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved: {RESULTS_CSV}")
    print(f"Saved: {REPORT_MD}")
    print("\n" + df_results.to_string(index=False))


if __name__ == "__main__":
    main()
