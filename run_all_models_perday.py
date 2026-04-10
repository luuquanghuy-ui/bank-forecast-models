"""
Per-day predictions for ALL models:
1. GARCH(1,1) - volatility model
2. NeuralProphet - deep learning
3. TFT - attention-based

Compare predictions on each day of test set.
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from arch import arch_model
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = BASE_DIR / "perday_analysis"
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


def split_data(df):
    n = len(df)
    return df.iloc[:int(n*0.70)].copy(), df.iloc[int(n*0.70):int(n*0.85)].copy(), df.iloc[int(n*0.85):].copy()


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

    for i in range(len(test_ret)):
        sigma2[i] = omega + alpha * (eps_last ** 2 if i > 0 else (test_ret[0] - mu) ** 2) + beta * sigma2_last
        if i > 0:
            eps = (test_ret[i-1] - mu) * 100.0
            sigma2[i] = omega + alpha * (eps ** 2) + beta * sigma2_last
        sigma2_last = sigma2[i]

    return np.sqrt(sigma2) / 100.0


def garch_walkforward_correct(train_ret, test_ret):
    """GARCH(1,1) walk-forward - correct implementation."""
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


def main():
    print("=" * 70)
    print("PER-DAY PREDICTIONS: GARCH vs Naive")
    print("=" * 70)

    all_results = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        df = load_data(path)
        train, val, test = split_data(df)

        train_ret = train["log_return"].values
        val_ret = val["log_return"].values
        test_ret = test["log_return"].values
        test_abs = np.abs(test_ret)

        # Combine train+val for GARCH fitting
        trainval_ret = np.concatenate([train_ret, val_ret])

        # GARCH predictions
        garch_sigma = garch_walkforward_correct(trainval_ret, test_ret)
        garch_pred = np.sqrt(2 / np.pi) * garch_sigma

        # Naive: predict 0 volatility
        naive_pred = np.zeros(len(test_ret))

        # Create per-day dataframe
        test_df = pd.DataFrame({
            "ds": test["ds"].values,
            "actual": test_abs,
            "naive_pred": naive_pred,
            "garch_pred": garch_pred,
            "naive_error": np.abs(test_abs - naive_pred),
            "garch_error": np.abs(test_abs - garch_pred)
        })

        # Save per-day predictions
        test_df.to_csv(OUTPUT_DIR / f"{bank}_perday.csv", index=False)

        # Summary stats
        naive_mae = mean_absolute_error(test_abs, naive_pred)
        garch_mae = mean_absolute_error(test_abs, garch_pred)

        print(f"  Test period: {test_df['ds'].min().strftime('%Y-%m-%d')} to {test_df['ds'].max().strftime('%Y-%m-%d')}")
        print(f"  Days: {len(test_df)}")
        print(f"  Naive MAE: {naive_mae:.6f}")
        print(f"  GARCH MAE: {garch_mae:.6f}")

        # Worst/best days
        worst_garch = test_df.nlargest(5, "garch_error")
        best_garch = test_df.nsmallest(5, "garch_error")

        print(f"\n  Worst 5 days (GARCH error):")
        for _, row in worst_garch.iterrows():
            print(f"    {row['ds'].strftime('%Y-%m-%d')}: actual={row['actual']:.4f}, pred={row['garch_pred']:.4f}, error={row['garch_error']:.4f}")

        print(f"\n  Best 5 days (GARCH error):")
        for _, row in best_garch.iterrows():
            print(f"    {row['ds'].strftime('%Y-%m-%d')}: actual={row['actual']:.4f}, pred={row['garch_pred']:.4f}, error={row['garch_error']:.4f}")

        # Save error stats
        all_results.append({
            "bank": bank,
            "n_days": len(test_df),
            "start_date": test_df['ds'].min().strftime('%Y-%m-%d'),
            "end_date": test_df['ds'].max().strftime('%Y-%m-%d'),
            "naive_mae": naive_mae,
            "garch_mae": garch_mae,
            "worst_date": worst_garch.iloc[0]['ds'].strftime('%Y-%m-%d'),
            "worst_error": worst_garch.iloc[0]['garch_error'],
            "best_date": best_garch.iloc[0]['ds'].strftime('%Y-%m-%d'),
            "best_error": best_garch.iloc[0]['garch_error'],
        })

        # Create chart
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        ax1 = axes[0]
        ax1.plot(test_df["ds"], test_df["actual"], label="Actual", color="black", linewidth=1, alpha=0.8)
        ax1.plot(test_df["ds"], test_df["garch_pred"], label="GARCH", color="red", linewidth=1, alpha=0.8)
        ax1.fill_between(test_df["ds"], test_df["actual"], test_df["garch_pred"], alpha=0.2, color="gray")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("|Log Return|")
        ax1.set_title(f"{bank} - GARCH Per-Day Predictions (Test Set)")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=30)

        ax2 = axes[1]
        ax2.bar(test_df["ds"], test_df["garch_error"], color="steelblue", alpha=0.7, width=1)
        ax2.axhline(y=garch_mae, color="red", linestyle="--", label=f"Mean ({garch_mae:.4f})")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Absolute Error")
        ax2.set_title(f"{bank} - GARCH Daily Prediction Error")
        ax2.legend()
        ax2.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{bank}_garch_perday.png", dpi=150)
        plt.close()
        print(f"\n  Chart saved: {bank}_garch_perday.png")

    # Save summary
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(OUTPUT_DIR / "perday_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for _, r in df_summary.iterrows():
        print(f"\n{r['bank']}: {r['n_days']} days ({r['start_date']} to {r['end_date']})")
        print(f"  Naive MAE: {r['naive_mae']:.6f}")
        print(f"  GARCH MAE: {r['garch_mae']:.6f}")
        print(f"  Worst: {r['worst_date']} (error={r['worst_error']:.4f})")
        print(f"  Best: {r['best_date']} (error={r['best_error']:.4f})")

    print(f"\n\nAll files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()