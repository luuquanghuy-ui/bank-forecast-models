"""
Per-Day Analysis for ALL Models:
1. GARCH (volatility)
2. Hybrid (volatility)
3. NeuralProphet (price)
4. TFT (price)

Compare all models fairly on both price and volatility.
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = BASE_DIR / "perday_all_models"
OUTPUT_DIR.mkdir(exist_ok=True)

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"date": "ds", "close": "price"})
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df["volatility"] = np.abs(df["log_return"])
    return df.dropna().reset_index(drop=True)


def main():
    print("=" * 70)
    print("PER-DAY ANALYSIS: ALL MODELS")
    print("=" * 70)

    all_results = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        # Load actual data
        df = load_data(path)
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        test_start = val_end
        test_end = n

        test_df = df.iloc[test_start:test_end].copy().reset_index(drop=True)
        print(f"Test period: {test_df['ds'].min().date()} to {test_df['ds'].max().date()}")
        print(f"Test days: {len(test_df)}")

        # Load GARCH per-day
        garch = pd.read_csv(f"perday_analysis/{bank}_perday.csv")
        garch["ds"] = pd.to_datetime(garch["ds"])
        garch = garch.rename(columns={"garch_pred": "garch_vol", "actual": "volatility_garch"})

        # Load Hybrid per-day
        hybrid = pd.read_csv(f"Hybrid_GARCH_DL/detailed_analysis/{bank}_daily_predictions.csv")
        hybrid["ds"] = pd.to_datetime(hybrid["ds"])
        hybrid = hybrid.rename(columns={"ensemble_pred": "hybrid_vol"})

        # Load NP per-day (price)
        np_pred = pd.read_csv(f"NeuralProphet/neuralprophet_main_outputs/{bank}_neuralprophet_test_forecast.csv")
        np_pred["ds"] = pd.to_datetime(np_pred["ds"])
        np_pred = np_pred.rename(columns={"yhat1": "np_price_pred"})

        # Load TFT per-day (price)
        tft_pred = pd.read_csv(f"TemporalFusionTransformer/tft_main_outputs/{bank}_tft_test_forecast.csv")
        tft_pred["ds"] = pd.to_datetime(tft_pred["ds"])
        tft_pred = tft_pred.rename(columns={"yhat1": "tft_price_pred"})

        # Merge all
        merged = test_df[["ds", "price", "volatility"]].copy()

        # GARCH/Hybrid
        merged = pd.merge(merged, garch[["ds", "garch_vol"]], on="ds", how="left")
        merged = pd.merge(merged, hybrid[["ds", "hybrid_vol"]], on="ds", how="left")

        # NP/TFT (price)
        merged = pd.merge(merged, np_pred[["ds", "np_price_pred"]], on="ds", how="left")
        merged = pd.merge(merged, tft_pred[["ds", "tft_price_pred"]], on="ds", how="left")

        # Drop NaN
        merged = merged.dropna().reset_index(drop=True)
        print(f"Valid days after merge: {len(merged)}")

        # === VOLATILITY PREDICTION ===
        naive_vol = np.zeros(len(merged))
        naive_vol_mae = mean_absolute_error(merged["volatility"], naive_vol)
        garch_vol_mae = mean_absolute_error(merged["volatility"], merged["garch_vol"])
        hybrid_vol_mae = mean_absolute_error(merged["volatility"], merged["hybrid_vol"])

        # NP/TFT volatility from price prediction
        merged["np_vol_from_price"] = np.abs(np.log(merged["np_price_pred"] / merged["np_price_pred"].shift(1)))
        merged["tft_vol_from_price"] = np.abs(np.log(merged["tft_price_pred"] / merged["tft_price_pred"].shift(1)))
        merged["np_vol_from_price"] = merged["np_vol_from_price"].fillna(0).replace([np.inf, -np.inf], 0)
        merged["tft_vol_from_price"] = merged["tft_vol_from_price"].fillna(0).replace([np.inf, -np.inf], 0)

        np_vol_mae = mean_absolute_error(merged["volatility"], merged["np_vol_from_price"])
        tft_vol_mae = mean_absolute_error(merged["volatility"], merged["tft_vol_from_price"])

        # === PRICE PREDICTION ===
        merged["naive_price"] = merged["price"].shift(1).fillna(merged["price"].iloc[0])
        naive_price_mae = mean_absolute_error(merged["price"], merged["naive_price"])
        np_price_mae = mean_absolute_error(merged["price"], merged["np_price_pred"])
        tft_price_mae = mean_absolute_error(merged["price"], merged["tft_price_pred"])

        # === SAVE PER-DAY CSV ===
        merged.to_csv(OUTPUT_DIR / f"{bank}_perday_all.csv", index=False)

        # === STATISTICS ===
        print(f"\n--- VOLATILITY PREDICTION ---")
        print(f"Naive (0):           MAE = {naive_vol_mae:.6f}")
        print(f"GARCH:               MAE = {garch_vol_mae:.6f} ({((naive_vol_mae-garch_vol_mae)/naive_vol_mae*100):.1f}%)")
        print(f"Hybrid:              MAE = {hybrid_vol_mae:.6f} ({((naive_vol_mae-hybrid_vol_mae)/naive_vol_mae*100):.1f}%)")
        print(f"NP (from price):    MAE = {np_vol_mae:.6f} ({((naive_vol_mae-np_vol_mae)/naive_vol_mae*100):.1f}%)")
        print(f"TFT (from price):   MAE = {tft_vol_mae:.6f} ({((naive_vol_mae-tft_vol_mae)/naive_vol_mae*100):.1f}%)")

        print(f"\n--- PRICE PREDICTION ---")
        print(f"Naive (last price): MAE = {naive_price_mae:.4f}")
        print(f"NP:                 MAE = {np_price_mae:.4f} ({((naive_price_mae-np_price_mae)/naive_price_mae*100):.1f}%)")
        print(f"TFT:                MAE = {tft_price_mae:.4f} ({((naive_price_mae-tft_price_mae)/naive_price_mae*100):.1f}%)")

        # Store summary
        all_results.append({
            "bank": bank,
            "n_days": len(merged),
            # Volatility
            "naive_vol_mae": naive_vol_mae,
            "garch_vol_mae": garch_vol_mae,
            "hybrid_vol_mae": hybrid_vol_mae,
            "np_vol_mae": np_vol_mae,
            "tft_vol_mae": tft_vol_mae,
            # Price
            "naive_price_mae": naive_price_mae,
            "np_price_mae": np_price_mae,
            "tft_price_mae": tft_price_mae,
            # Best
            "best_vol": "Hybrid" if hybrid_vol_mae < garch_vol_mae else "GARCH",
            "best_price": "Naive",
        })

        # === CREATE CHARTS ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Chart 1: Volatility prediction comparison
        ax1 = axes[0, 0]
        ax1.plot(merged["ds"], merged["volatility"], "k-", alpha=0.7, label="Actual", linewidth=0.8)
        ax1.plot(merged["ds"], merged["garch_vol"], "b-", alpha=0.6, label="GARCH", linewidth=0.8)
        ax1.plot(merged["ds"], merged["hybrid_vol"], "r-", alpha=0.6, label="Hybrid", linewidth=0.8)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("|Log Return|")
        ax1.set_title(f"{bank} - Volatility Prediction: GARCH vs Hybrid vs Actual")
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.tick_params(axis="x", rotation=45)

        # Chart 2: Price prediction comparison (NP/TFT vs Naive)
        ax2 = axes[0, 1]
        ax2.plot(merged["ds"], merged["price"], "k-", alpha=0.7, label="Actual", linewidth=0.8)
        ax2.plot(merged["ds"], merged["naive_price"], "g--", alpha=0.6, label="Naive", linewidth=0.8)
        ax2.plot(merged["ds"], merged["np_price_pred"], "orange", alpha=0.6, label="NP", linewidth=0.5)
        ax2.plot(merged["ds"], merged["tft_price_pred"], "purple", alpha=0.6, label="TFT", linewidth=0.5)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.set_title(f"{bank} - Price Prediction: NP/TFT vs Naive")
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.tick_params(axis="x", rotation=45)

        # Chart 3: Volatility MAE comparison bar chart
        ax3 = axes[1, 0]
        models = ["Naive", "GARCH", "Hybrid", "NP", "TFT"]
        maes = [naive_vol_mae, garch_vol_mae, hybrid_vol_mae, np_vol_mae, tft_vol_mae]
        colors = ["gray", "blue", "red", "orange", "purple"]
        bars = ax3.bar(models, maes, color=colors)
        ax3.set_ylabel("MAE")
        ax3.set_title(f"{bank} - Volatility MAE Comparison")
        for bar, mae in zip(bars, maes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                    f"{mae:.4f}", ha="center", va="bottom", fontsize=9)

        # Chart 4: Price MAE comparison bar chart
        ax4 = axes[1, 1]
        models_p = ["Naive", "NP", "TFT"]
        maes_p = [naive_price_mae, np_price_mae, tft_price_mae]
        colors_p = ["gray", "orange", "purple"]
        bars_p = ax4.bar(models_p, maes_p, color=colors_p)
        ax4.set_ylabel("MAE")
        ax4.set_title(f"{bank} - Price MAE Comparison")
        for bar, mae in zip(bars_p, maes_p):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{mae:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{bank}_all_models_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved: {bank}_all_models_comparison.png")

    # Save summary
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(OUTPUT_DIR / "perday_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY: ALL MODELS PER-DAY ANALYSIS")
    print("=" * 70)

    for _, r in df_summary.iterrows():
        print(f"\n{r['bank']} ({r['n_days']} days):")
        print(f"  VOLATILITY: Naive={r['naive_vol_mae']:.4f}, GARCH={r['garch_vol_mae']:.4f}, Hybrid={r['hybrid_vol_mae']:.4f}")
        print(f"  PRICE: Naive={r['naive_price_mae']:.4f}, NP={r['np_price_mae']:.4f}, TFT={r['tft_price_mae']:.4f}")
        print(f"  BEST VOLATILITY: {r['best_vol']}")
        print(f"  BEST PRICE: {r['best_price']}")

    print(f"\n\nSaved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()