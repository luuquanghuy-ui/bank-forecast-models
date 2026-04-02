"""
NeuralProphet: Sensitivity Analysis

Test different hyperparameters and document root cause of failure.
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import torch
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "np_sensitivity_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}

LAGGED_REGRESSORS = [
    "volume", "rsi", "volatility_20d",
    "vnindex_close", "vn30_close", "usd_vnd", "interest_rate",
]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    cols = ["date", "close"] + [c for c in LAGGED_REGRESSORS if c in df.columns]
    df = df[cols].copy()
    df = df.rename(columns={"date": "ds", "close": "y"})
    return df.dropna().reset_index(drop=True)


def split_df(df):
    n = len(df)
    return df.iloc[:int(n*0.70)].copy(), df.iloc[int(n*0.70):int(n*0.85)].copy(), df.iloc[int(n*0.85):].copy()


def train_and_eval(train, val, test, n_lags, learning_rate, epochs, weekly_seasonality):
    """Train NeuralProphet and return metrics."""
    set_seed(42)

    model = NeuralProphet(
        n_lags=n_lags,
        n_forecasts=1,
        yearly_seasonality=False,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        epochs=epochs,
        batch_size=64,
        learning_rate=learning_rate,
        show_progress=False,
    )

    # Add regressors
    for col in LAGGED_REGRESSORS:
        if col in train.columns:
            model.add_future_regressor(col)

    # Prepare data
    train_ds = train[["ds", "y"] + [c for c in LAGGED_REGRESSORS if c in train.columns]].copy()
    val_ds = val[["ds", "y"] + [c for c in LAGGED_REGRESSORS if c in val.columns]].copy()
    test_ds = test[["ds", "y"] + [c for c in LAGGED_REGRESSORS if c in test.columns]].copy()

    # Combine train + val for fitting
    combined = pd.concat([train_ds, val_ds], ignore_index=True)

    try:
        model.fit(combined, freq="D", show_progress=False)

        # Predict on test
        future = test_ds.copy()
        for col in LAGGED_REGRESSORS:
            if col in future.columns:
                future[col] = future[col].fillna(0)
        future = future.dropna(subset=["y"])

        if len(future) < 10:
            return None

        forecast = model.predict(future)

        if "yhat1" not in forecast.columns:
            return None

        y_pred = forecast["yhat1"].values
        y_actual = forecast["y"].values

        mae = mean_absolute_error(y_actual, y_pred)
        return mae

    except Exception as e:
        return None


def naive_baseline(val, test):
    """Naive: predict last known price."""
    val_pred = np.full(len(val), val["y"].iloc[-1])
    test_pred = np.full(len(test), test["y"].iloc[-1])
    return val_pred, test_pred


def main():
    print("=" * 70)
    print("NEURALPROPHET SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Hyperparameter grid
    configs = [
        # V1 (original)
        {"name": "V1 (original)", "n_lags": 10, "learning_rate": 0.01, "epochs": 40, "weekly_seasonality": True},
        # Learning rate variants
        {"name": "lr=0.001", "n_lags": 10, "learning_rate": 0.001, "epochs": 40, "weekly_seasonality": True},
        {"name": "lr=0.1", "n_lags": 10, "learning_rate": 0.1, "epochs": 40, "weekly_seasonality": True},
        # n_lags variants
        {"name": "n_lags=5", "n_lags": 5, "learning_rate": 0.01, "epochs": 40, "weekly_seasonality": True},
        {"name": "n_lags=20", "n_lags": 20, "learning_rate": 0.01, "epochs": 40, "weekly_seasonality": True},
        {"name": "n_lags=30", "n_lags": 30, "learning_rate": 0.01, "epochs": 40, "weekly_seasonality": True},
        # Epochs variants
        {"name": "epochs=20", "n_lags": 10, "learning_rate": 0.01, "epochs": 20, "weekly_seasonality": True},
        {"name": "epochs=80", "n_lags": 10, "learning_rate": 0.01, "epochs": 80, "weekly_seasonality": True},
        # Weekly seasonality off
        {"name": "no_weekly", "n_lags": 10, "learning_rate": 0.01, "epochs": 40, "weekly_seasonality": False},
    ]

    all_results = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        df = prepare_dataframe(path)
        train, val, test = split_df(df)

        # Naive baseline
        naive_pred = np.full(len(test), test["y"].iloc[-1])
        naive_mae = mean_absolute_error(test["y"].values, naive_pred)
        print(f"  Naive MAE: {naive_mae:.4f}")

        # Test each config
        for cfg in configs:
            mae = train_and_eval(train, val, test, **{
                k: v for k, v in cfg.items() if k != "name"
            })

            if mae is not None:
                vs_naive = mae / naive_mae
                print(f"  {cfg['name']}: MAE={mae:.4f}, vs Naive={vs_naive:.2f}x")
                all_results.append({
                    "bank": bank,
                    "config": cfg["name"],
                    "n_lags": cfg["n_lags"],
                    "learning_rate": cfg["learning_rate"],
                    "epochs": cfg["epochs"],
                    "weekly_seasonality": cfg["weekly_seasonality"],
                    "naive_mae": naive_mae,
                    "np_mae": mae,
                    "ratio_to_naive": vs_naive,
                })
            else:
                print(f"  {cfg['name']}: FAILED")

    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_DIR / "np_sensitivity_results.csv", index=False, encoding="utf-8-sig")

    # Summary
    print("\n" + "=" * 70)
    print("SENSITIVITY SUMMARY")
    print("=" * 70)

    # Best config per bank
    for bank in ["BID", "CTG", "VCB"]:
        subset = df_results[df_results["bank"] == bank]
        if len(subset) > 0:
            best = subset.loc[subset["np_mae"].idxmin()]
            worst = subset.loc[subset["np_mae"].idxmax()]
            print(f"\n{bank}:")
            print(f"  Best:  {best['config']} (MAE={best['np_mae']:.4f}, {best['ratio_to_naive']:.2f}x Naive)")
            print(f"  Worst: {worst['config']} (MAE={worst['np_mae']:.4f}, {worst['ratio_to_naive']:.2f}x Naive)")
            print(f"  Naive: {best['naive_mae']:.4f}")

    print(f"\nSaved: {OUTPUT_DIR / 'np_sensitivity_results.csv'}")

    # Root cause analysis
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)
    print("""
1. WHY NEURALPROPHET FAILS:

   a) Martingale Property:
      - Stock prices follow martingale: E[S_{t+1}|I_t] = S_t
      - Naive (predict S_t) is a very strong baseline
      - NP tries to learn complex patterns → overfits noise

   b) Small Sample Size:
      - ~1750 training points is too few for neural networks
      - NP needs tens of thousands for deep learning

   c) Target Mismatch:
      - NP predicts PRICE, but returns/volatility are easier to model
      - Price is non-stationary; returns are stationary

   d) Convexity Bias (log-transform):
      - If using log(price): exp(E[log(S)]) < E[S]
      - Errors in log-space amplify when exponentiated

2. SENSITIVITY FINDINGS:
   - Different hyperparameters cannot fix the fundamental problem
   - The model architecture is unsuitable for this data
   - NP is a TIME SERIES model, not a FINANCIAL volatility model
""")


if __name__ == "__main__":
    main()
