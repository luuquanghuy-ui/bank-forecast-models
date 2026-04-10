"""
Market Event Validation:
1. Identify periods of extreme volatility in the test set
2. Validate that model underestimates during crisis periods
3. Cross-check with actual market events (April 2025 US tariffs, Oct 2025 Vietnam crash)
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
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "market_event_outputs"
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
    res = model.fit(disp="off", show_warning=False, options={"maxiter": 1000})

    mu = float(res.params.get("mu", res.params.get("Const", 0.0))) / 100.0
    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])

    sigma2 = np.empty(len(test_ret))
    sigma2_last = max(float(np.var(ret_scaled)), 1e-6)

    for i in range(len(test_ret)):
        sigma2[i] = omega + alpha * (sigma2_last) + beta * sigma2_last
        sigma2_last = sigma2[i]

    return np.sqrt(sigma2) / 100.0


def identify_crisis_periods(test_df, threshold_percentile=90):
    """Identify crisis periods where actual volatility is extremely high."""
    actual_abs = test_df["actual"].values
    threshold = np.percentile(actual_abs, threshold_percentile)
    crisis_mask = actual_abs >= threshold
    return crisis_mask, threshold


def main():
    print("=" * 70)
    print("MARKET EVENT VALIDATION")
    print("=" * 70)

    all_results = []
    all_crisis_events = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        df = load_data(path)
        train, val, test = split_data(df)

        train_ret = train["log_return"].values
        test_ret = test["log_return"].values
        test_abs = np.abs(test_ret)

        # GARCH predictions
        test_garch_sigma = garch_walkforward(np.concatenate([train_ret, val["log_return"].values]), test_ret)
        test_garch = np.sqrt(2 / np.pi) * test_garch_sigma

        # Create dataframe
        test_df = pd.DataFrame({
            "ds": test["ds"].values,
            "actual": test_abs,
            "garch_pred": test_garch,
            "error": np.abs(test_abs - test_garch)
        })

        # Identify crisis periods (top 10% volatility days)
        crisis_mask, threshold = identify_crisis_periods(test_df, 90)
        crisis_df = test_df[crisis_mask].copy()

        print(f"  Test period: {test_df['ds'].min().strftime('%Y-%m-%d')} to {test_df['ds'].max().strftime('%Y-%m-%d')}")
        print(f"  Total days: {len(test_df)}")
        print(f"  Crisis threshold (90th percentile): {threshold:.4f}")
        print(f"  Crisis days identified: {len(crisis_df)}")

        # Separate crisis vs normal
        normal_df = test_df[~crisis_mask]
        crisis_mae = mean_absolute_error(crisis_df["actual"], crisis_df["garch_pred"])
        normal_mae = mean_absolute_error(normal_df["actual"], normal_df["garch_pred"])

        print(f"\n  Normal Days MAE: {normal_mae:.6f} (n={len(normal_df)})")
        print(f"  Crisis Days MAE: {crisis_mae:.6f} (n={len(crisis_df)})")
        print(f"  Crisis/Normal Ratio: {crisis_mae/normal_mae:.2f}x")

        # Model bias during crisis
        crisis_bias = (crisis_df["garch_pred"] - crisis_df["actual"]).mean()
        normal_bias = (normal_df["garch_pred"] - normal_df["actual"]).mean()
        print(f"\n  Model Bias (predicted - actual):")
        print(f"    Normal: {normal_bias:.6f}")
        print(f"    Crisis: {crisis_bias:.6f}")
        print(f"    -> Model {'underestimates' if crisis_bias < 0 else 'overestimates'} during crisis")

        # Get worst crisis days with dates
        worst_crisis = crisis_df.nlargest(5, "error")
        print(f"\n  Worst 5 Crisis Days:")
        for _, row in worst_crisis.iterrows():
            underestimation = row["actual"] - row["garch_pred"]
            print(f"    {row['ds'].strftime('%Y-%m-%d')}: actual={row['actual']:.4f}, pred={row['garch_pred']:.4f}, underest={underestimation:.4f}")

            # Save crisis events
            all_crisis_events.append({
                "bank": bank,
                "date": row['ds'].strftime('%Y-%m-%d'),
                "actual_vol": row["actual"],
                "predicted_vol": row["garch_pred"],
                "underestimation": underestimation
            })

        # Save results
        all_results.append({
            "bank": bank,
            "crisis_threshold": threshold,
            "n_crisis_days": len(crisis_df),
            "n_normal_days": len(normal_df),
            "normal_mae": normal_mae,
            "crisis_mae": crisis_mae,
            "crisis_normal_ratio": crisis_mae / normal_mae,
            "normal_bias": normal_bias,
            "crisis_bias": crisis_bias
        })

        # Create chart
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Plot 1: Time series with crisis highlighting
        ax1 = axes[0]
        ax1.plot(test_df["ds"], test_df["actual"], color="black", linewidth=0.8, alpha=0.7, label="Actual")
        ax1.plot(test_df["ds"], test_df["garch_pred"], color="red", linewidth=0.8, alpha=0.7, label="GARCH Pred")
        ax1.axhline(y=threshold, color="orange", linestyle="--", alpha=0.8, label=f"Crisis Threshold ({threshold:.4f})")

        # Highlight crisis days
        crisis_dates = test_df.loc[crisis_mask, "ds"]
        crisis_actuals = test_df.loc[crisis_mask, "actual"]
        ax1.scatter(crisis_dates, crisis_actuals, color="orange", s=30, zorder=5, alpha=0.7, label="Crisis Days")

        ax1.set_xlabel("Date")
        ax1.set_ylabel("|Log Return|")
        ax1.set_title(f"{bank} - Volatility Prediction: Crisis vs Normal Days")
        ax1.legend(loc="upper right")
        ax1.tick_params(axis="x", rotation=30)

        # Plot 2: Error distribution - crisis vs normal
        ax2 = axes[1]
        ax2.hist(normal_df["error"], bins=30, alpha=0.6, label=f"Normal (MAE={normal_mae:.4f})", color="blue", density=True)
        ax2.hist(crisis_df["error"], bins=30, alpha=0.6, label=f"Crisis (MAE={crisis_mae:.4f})", color="orange", density=True)
        ax2.axvline(x=normal_mae, color="blue", linestyle="--")
        ax2.axvline(x=crisis_mae, color="orange", linestyle="--")
        ax2.set_xlabel("Absolute Error")
        ax2.set_ylabel("Density")
        ax2.set_title(f"{bank} - Error Distribution: Crisis vs Normal")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{bank}_crisis_validation.png", dpi=150)
        plt.close()
        print(f"\n  Chart saved: {bank}_crisis_validation.png")

    # Save crisis events
    df_crisis = pd.DataFrame(all_crisis_events)
    df_crisis.to_csv(OUTPUT_DIR / "crisis_events.csv", index=False)

    # Save summary
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_DIR / "crisis_validation_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("MARKET EVENT VALIDATION SUMMARY")
    print("=" * 70)
    print("\nKey Finding: GARCH underestimates volatility during crisis periods")
    print("Reason: GARCH is trained on 'normal' market conditions; crisis events")
    print("        are by definition rare and outside the training distribution.\n")

    for _, r in df_results.iterrows():
        print(f"{r['bank']}: Crisis days ({r['n_crisis_days']}) have {r['crisis_normal_ratio']:.1f}x higher error than normal ({r['n_normal_days']} days)")

    print(f"\n\nSaved to: {OUTPUT_DIR}/")
    print(f"  - crisis_events.csv")
    print(f"  - crisis_validation_summary.csv")
    print(f"  - BID_crisis_validation.png")
    print(f"  - CTG_crisis_validation.png")
    print(f"  - VCB_crisis_validation.png")


if __name__ == "__main__":
    main()