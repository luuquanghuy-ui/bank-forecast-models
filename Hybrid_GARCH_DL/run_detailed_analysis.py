"""
Detailed per-day analysis for thesis:
1. Per-day predictions with dates
2. Error analysis (worst/best days)
3. Actual vs Predicted charts
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from arch import arch_model
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "detailed_analysis"
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
    sigma2_last = max(float(np.var(ret_scaled)), 1e-8)
    eps_last = (test_ret[0] - mu) * 100.0

    for i in range(len(test_ret)):
        sigma2[i] = omega + alpha * (eps_last ** 2) + beta * sigma2_last
        eps_last = (test_ret[i] - mu) * 100.0
        sigma2_last = sigma2[i]

    return np.sqrt(sigma2) / 100.0


def make_features(data, available):
    X = data[available].values
    for i in range(1, 6):
        X = np.column_stack([X, data["log_return"].shift(i).fillna(0).values])
    return X


def main():
    features = ["volume", "rsi", "volatility_20d", "vnindex_close", "vn30_close", "usd_vnd", "interest_rate"]
    best_weights = {"BID": 0.45, "CTG": 0.50, "VCB": 0.40}

    all_error_stats = []
    all_predictions = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*60}\n{bank}\n{'='*60}")

        df = load_data(path)
        train, val, test = split_data(df)

        train_ret = train["log_return"].values
        test_ret = test["log_return"].values
        test_abs = np.abs(test_ret)

        available = [f for f in features if f in df.columns]
        X_train = make_features(train, available)
        X_test = make_features(test, available)

        # GARCH predictions
        test_garch_sigma = garch_walkforward(np.concatenate([train_ret, val["log_return"].values]), test_ret)
        test_garch = np.sqrt(2 / np.pi) * test_garch_sigma

        # Ridge predictions
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, train_ret)
        test_ridge = np.abs(ridge.predict(X_test))

        # Ensemble
        w = best_weights[bank]
        test_ensemble = w * test_garch + (1-w) * test_ridge

        # Create per-day dataframe
        test_df = test.copy().reset_index(drop=True)
        test_df["actual"] = test_abs
        test_df["garch_pred"] = test_garch
        test_df["ridge_pred"] = test_ridge
        test_df["ensemble_pred"] = test_ensemble
        test_df["garch_error"] = np.abs(test_df["actual"] - test_df["garch_pred"])
        test_df["ridge_error"] = np.abs(test_df["actual"] - test_df["ridge_pred"])
        test_df["ensemble_error"] = np.abs(test_df["actual"] - test_df["ensemble_pred"])

        # Save per-day predictions
        pred_cols = ["ds", "actual", "garch_pred", "ridge_pred", "ensemble_pred",
                     "garch_error", "ridge_error", "ensemble_error"]
        test_df[pred_cols].to_csv(OUTPUT_DIR / f"{bank}_daily_predictions.csv", index=False)
        all_predictions.append(test_df[pred_cols])

        print(f"\n  Test period: {test_df['ds'].min().strftime('%Y-%m-%d')} to {test_df['ds'].max().strftime('%Y-%m-%d')}")
        print(f"  Total days: {len(test_df)}")

        # Error analysis
        worst_5 = test_df.nlargest(5, "ensemble_error")
        best_5 = test_df.nsmallest(5, "ensemble_error")

        print(f"\n  WORST 5 days (Highest Error):")
        for _, row in worst_5.iterrows():
            print(f"    {row['ds'].strftime('%Y-%m-%d')}: actual={row['actual']:.4f}, pred={row['ensemble_pred']:.4f}, error={row['ensemble_error']:.4f}")

        print(f"\n  BEST 5 days (Lowest Error):")
        for _, row in best_5.iterrows():
            print(f"    {row['ds'].strftime('%Y-%m-%d')}: actual={row['actual']:.4f}, pred={row['ensemble_pred']:.4f}, error={row['ensemble_error']:.4f}")

        # Statistics
        print(f"\n  Error Statistics:")
        print(f"    Mean: {test_df['ensemble_error'].mean():.4f}")
        print(f"    Std:  {test_df['ensemble_error'].std():.4f}")
        print(f"    Min:  {test_df['ensemble_error'].min():.4f}")
        print(f"    Max:  {test_df['ensemble_error'].max():.4f}")
        print(f"    Median: {test_df['ensemble_error'].median():.4f}")

        # Save error stats
        all_error_stats.append({
            "bank": bank,
            "n_days": len(test_df),
            "start_date": test_df['ds'].min().strftime('%Y-%m-%d'),
            "end_date": test_df['ds'].max().strftime('%Y-%m-%d'),
            "mean_error": test_df['ensemble_error'].mean(),
            "std_error": test_df['ensemble_error'].std(),
            "worst_date": worst_5.iloc[0]['ds'].strftime('%Y-%m-%d'),
            "worst_error": worst_5.iloc[0]['ensemble_error'],
            "best_date": best_5.iloc[0]['ds'].strftime('%Y-%m-%d'),
            "best_error": best_5.iloc[0]['ensemble_error'],
        })

        # Create charts
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Plot 1: Actual vs Predicted
        ax1 = axes[0]
        ax1.plot(test_df["ds"], test_df["actual"], label="Actual", color="black", linewidth=1, alpha=0.8)
        ax1.plot(test_df["ds"], test_df["ensemble_pred"], label="Ensemble", color="red", linewidth=1, alpha=0.8)
        ax1.fill_between(test_df["ds"], test_df["actual"], test_df["ensemble_pred"], alpha=0.2, color="gray")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("|Log Return|")
        ax1.set_title(f"{bank} - Actual vs Ensemble Predictions (Test Set)")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=30)

        # Plot 2: Daily Error
        ax2 = axes[1]
        ax2.bar(test_df["ds"], test_df["ensemble_error"], color="steelblue", alpha=0.7, width=1)
        ax2.axhline(y=test_df["ensemble_error"].mean(), color="red", linestyle="--",
                    label=f"Mean ({test_df['ensemble_error'].mean():.4f})")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Absolute Error")
        ax2.set_title(f"{bank} - Daily Prediction Error")
        ax2.legend()
        ax2.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{bank}_actual_vs_predicted.png", dpi=150)
        plt.close()
        print(f"\n  Chart saved: {bank}_actual_vs_predicted.png")

        # Feature importance
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, train_ret)
        feat_imp = pd.DataFrame({
            "feature": available + [f"lag_{i}" for i in range(1, 6)],
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)

        print(f"\n  Feature Importance (RF):")
        for _, row in feat_imp.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

        feat_imp.to_csv(OUTPUT_DIR / f"{bank}_feature_importance.csv", index=False)

    # Save summary
    df_stats = pd.DataFrame(all_error_stats)
    df_stats.to_csv(OUTPUT_DIR / "error_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for _, r in df_stats.iterrows():
        print(f"\n{r['bank']}: {r['n_days']} days ({r['start_date']} to {r['end_date']})")
        print(f"  Mean error: {r['mean_error']:.4f} +/- {r['std_error']:.4f}")
        print(f"  Worst: {r['worst_date']} (error={r['worst_error']:.4f})")
        print(f"  Best: {r['best_date']} (error={r['best_error']:.4f})")

    print(f"\n\nAll files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
