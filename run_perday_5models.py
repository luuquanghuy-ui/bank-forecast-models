"""
Per-Day Analysis: 5 Models for Thesis

Models (5 total):
- Naive (baseline)
- XGBoost (core model 1)
- NeuralProphet (NP) (core model 2)
- TFT (core model 3)
- Hybrid = GARCH volatility signal + Ridge (developed from thesis)

Note: GARCH and Ridge are ONLY used within Hybrid, not as standalone models.
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from arch import arch_model
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# TFT imports
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from lightning.pytorch import Trainer

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

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
    df = df.rename(columns={"date": "ds"})
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = np.abs(df["log_return"])
    return df.dropna().reset_index(drop=True)


def create_features(df):
    data = df.copy()
    data["volume_lag1"] = data["volume"].shift(1)
    data["volatility_5d"] = data["log_return"].rolling(5).std().shift(1)
    data["volatility_20d"] = data["log_return"].rolling(20).std().shift(1)
    data["rsi_lag1"] = data["rsi"].shift(1)
    data["return_lag1"] = data["log_return"].shift(1)
    data["return_lag2"] = data["log_return"].shift(2)
    data["return_lag5"] = data["log_return"].shift(5)
    data["price_lag1"] = data["close"].shift(1)
    data["price_ma5"] = data["close"].rolling(5).mean().shift(1)
    return data.dropna().reset_index(drop=True)


# ===== NAIVE BASELINE =====

def naive_vol_walkforward(test_ret):
    """Naive baseline: predict 0 for volatility"""
    return np.zeros(len(test_ret))


def naive_price_walkforward(test_df):
    """Naive baseline: predict last known price"""
    prices = test_df["close"].values
    pred = np.zeros(len(prices))
    pred[1:] = prices[:-1]
    return pred


# ===== XGBOOST (CORE MODEL 1) =====

def xgboost_vol_walkforward(train_df, test_df):
    """XGBoost for volatility"""
    feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["volatility"].values
    X_test = test_df[feature_cols].values
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def xgboost_price_walkforward(train_df, test_df):
    """XGBoost for price"""
    feature_cols = ["volume_lag1", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5", "price_lag1", "price_ma5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["close"].values
    X_test = test_df[feature_cols].values
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return np.nan_to_num(pred, nan=test_df["close"].iloc[0])


# ===== NEURALPROPHET (CORE MODEL 2) =====

def np_vol_walkforward(train_df, test_df):
    """NeuralProphet for volatility"""
    train_prophet = train_df[["ds", "volatility"]].copy()
    train_prophet.columns = ["ds", "y"]
    val_prophet = test_df[["ds", "volatility"]].copy()
    val_prophet.columns = ["ds", "y"]

    model = NeuralProphet(learning_rate=0.01, epochs=15, n_lags=10, n_forecasts=1, loss_func="MAE", weekly_seasonality=False)
    model.fit(train_prophet, freq="D", validation_df=val_prophet)
    predictions = model.predict(val_prophet)

    pred_values = predictions["yhat1"].values
    pred_values = np.nan_to_num(pred_values, nan=0.0)
    return pred_values


def np_price_walkforward(train_df, test_df):
    """NeuralProphet for price"""
    train_prophet = train_df[["ds", "close"]].copy()
    train_prophet.columns = ["ds", "y"]
    val_prophet = test_df[["ds", "close"]].copy()
    val_prophet.columns = ["ds", "y"]

    model = NeuralProphet(learning_rate=0.01, epochs=15, n_lags=10, n_forecasts=1, loss_func="MAE", weekly_seasonality=False)
    model.fit(train_prophet, freq="D", validation_df=val_prophet)
    predictions = model.predict(val_prophet)

    pred_values = predictions["yhat1"].values
    pred_values = np.nan_to_num(pred_values, nan=test_df["close"].iloc[0])
    return pred_values


# ===== TFT (CORE MODEL 3) =====

def tft_vol_walkforward(train_df, test_df, bank="UNKNOWN"):
    """TFT for volatility prediction"""
    lookback = 24
    train_d = train_df.copy()
    test_d = test_df.copy()

    train_d["time_idx"] = range(len(train_d))
    test_d["time_idx"] = range(len(train_d), len(train_d) + len(test_d))
    train_d["bank"] = bank
    test_d["bank"] = bank

    training = TimeSeriesDataSet(
        train_d,
        time_idx="time_idx",
        target="volatility",
        group_ids=["bank"],
        max_encoder_length=lookback,
        max_prediction_length=1,
        static_categoricals=["bank"],
        time_varying_unknown_reals=["volatility"],
        scalers={},
    )

    train_dataloader = training.to_dataloader(batch_size=64, num_workers=0, shuffle=False)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        learning_rate=0.001,
        optimizer="adam",
    )

    trainer = Trainer(
        max_epochs=15,
        accelerator="cpu",
        enable_progress_bar=False,
        logger=False,
    )

    trainer.fit(tft, train_dataloader)

    train_tail = train_d.iloc[-lookback:].copy()
    pred_df = pd.concat([train_tail, test_d], ignore_index=True).copy()
    pred_df = pred_df.reset_index(drop=True)

    pred_dataset = TimeSeriesDataSet.from_dataset(training, pred_df, predict=False)
    pred_dataloader = pred_dataset.to_dataloader(batch_size=64, num_workers=0, shuffle=False)

    raw_preds = tft.predict(pred_dataloader, mode="raw", return_x=True)
    # FIX: Index 3 = median (quantile 0.5), not index 4 (quantile 0.75)
    all_preds = raw_preds[0].prediction.cpu().numpy()[:, 0, 3].tolist()

    return np.array(all_preds)


def tft_price_walkforward(train_df, test_df, bank="UNKNOWN"):
    """TFT for price prediction"""
    lookback = 24
    train_d = train_df.copy()
    test_d = test_df.copy()

    train_d["time_idx"] = range(len(train_d))
    test_d["time_idx"] = range(len(train_d), len(train_d) + len(test_d))
    train_d["bank"] = bank
    test_d["bank"] = bank

    training = TimeSeriesDataSet(
        train_d,
        time_idx="time_idx",
        target="close",
        group_ids=["bank"],
        max_encoder_length=lookback,
        max_prediction_length=1,
        static_categoricals=["bank"],
        time_varying_unknown_reals=["close"],
        scalers={},
    )

    train_dataloader = training.to_dataloader(batch_size=64, num_workers=0, shuffle=False)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        learning_rate=0.001,
        optimizer="adam",
    )

    trainer = Trainer(
        max_epochs=15,
        accelerator="cpu",
        enable_progress_bar=False,
        logger=False,
    )

    trainer.fit(tft, train_dataloader)

    train_tail = train_d.iloc[-lookback:].copy()
    pred_df = pd.concat([train_tail, test_d], ignore_index=True).copy()
    pred_df = pred_df.reset_index(drop=True)

    pred_dataset = TimeSeriesDataSet.from_dataset(training, pred_df, predict=False)
    pred_dataloader = pred_dataset.to_dataloader(batch_size=64, num_workers=0, shuffle=False)

    raw_preds = tft.predict(pred_dataloader, mode="raw", return_x=True)
    # FIX: Index 3 = median (quantile 0.5), not index 4 (quantile 0.75)
    all_preds = raw_preds[0].prediction.cpu().numpy()[:, 0, 3].tolist()

    return np.array(np.nan_to_num(all_preds, nan=test_df["close"].iloc[0]))


# ===== HYBRID (DEVELOPED MODEL: GARCH + Ridge) =====

def garch_walkforward(train_ret, test_ret):
    """GARCH(1,1) for volatility signal - used within Hybrid"""
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


def hybrid_vol_walkforward(train_df, test_df, train_ret, test_ret, garch_weight=0.5):
    """Hybrid for volatility: GARCH volatility signal + Ridge"""
    garch_pred = garch_walkforward(train_ret, test_ret)
    feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["volatility"].values
    X_test = test_df[feature_cols].values
    train_vol = np.abs(train_ret)
    X_train_with_vol = np.column_stack([X_train, train_vol])
    X_test_with_vol = np.column_stack([X_test, garch_pred])
    model = Ridge(alpha=1.0)
    model.fit(X_train_with_vol, y_train)
    ridge_pred = model.predict(X_test_with_vol)
    return garch_weight * garch_pred + (1 - garch_weight) * ridge_pred


def hybrid_price_walkforward(train_df, test_df, train_ret, test_ret, garch_weight=0.3):
    """Hybrid for price: GARCH volatility signal + Ridge"""
    garch_vol_test = garch_walkforward(train_ret, test_ret)
    train_vol = np.abs(train_ret)
    feature_cols = ["volume_lag1", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5", "price_lag1", "price_ma5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["close"].values
    X_test = test_df[feature_cols].values
    X_train_with_vol = np.column_stack([X_train, train_vol])
    X_test_with_vol = np.column_stack([X_test, garch_vol_test])
    model = Ridge(alpha=1.0)
    model.fit(X_train_with_vol, y_train)
    pred = model.predict(X_test_with_vol)
    return np.nan_to_num(pred, nan=test_df["close"].iloc[0])


# ===== MAIN =====

def main():
    print("=" * 70)
    print("PER-DAY ANALYSIS: 5 MODELS FOR THESIS")
    print("=" * 70)
    print("Models: Naive, XGBoost, NP, TFT, Hybrid")
    print("Note: GARCH/Ridge only used within Hybrid (not standalone)")
    print("=" * 70)

    all_results = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        df = load_data(path)
        df = create_features(df)
        n = len(df)

        # Single split: 70% train, 30% test
        train_end = int(n * 0.70)
        test_start = train_end
        test_end = n

        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        test_df = df.iloc[test_start:test_end].copy().reset_index(drop=True)

        train_ret = train_df["log_return"].values
        test_ret = test_df["log_return"].values

        print(f"Train: {train_df['ds'].min().date()} to {train_df['ds'].max().date()} ({len(train_df)} days)")
        print(f"Test:  {test_df['ds'].min().date()} to {test_df['ds'].max().date()} ({len(test_df)} days)")

        # Initialize results dict
        results = {
            "ds": test_df["ds"].values,
            "actual_price": test_df["close"].values,
            "actual_volatility": test_df["volatility"].values,
        }

        # ===== VOLATILITY PREDICTIONS =====
        print("\n--- VOLATILITY ---")

        # Naive
        naive_vol_pred = naive_vol_walkforward(test_ret)
        naive_vol_mae = mean_absolute_error(np.abs(test_ret), naive_vol_pred)
        results["naive_vol_pred"] = naive_vol_pred
        print(f"  Naive:     MAE = {naive_vol_mae:.6f}")

        # XGBoost
        try:
            xgb_vol_pred = xgboost_vol_walkforward(train_df, test_df)
            xgb_vol_mae = mean_absolute_error(np.abs(test_ret), xgb_vol_pred)
            results["xgb_vol_pred"] = xgb_vol_pred
            print(f"  XGBoost:   MAE = {xgb_vol_mae:.6f}")
        except Exception as e:
            print(f"  XGBoost:   ERROR - {e}")
            xgb_vol_mae = np.nan

        # NeuralProphet
        try:
            np_vol_pred = np_vol_walkforward(train_df, test_df)
            np_vol_pred = np_vol_pred[:len(test_ret)]
            np_vol_mae = mean_absolute_error(np.abs(test_ret), np_vol_pred)
            results["np_vol_pred"] = np_vol_pred
            print(f"  NP:        MAE = {np_vol_mae:.6f}")
        except Exception as e:
            print(f"  NP:        ERROR - {e}")
            np_vol_mae = np.nan

        # TFT
        try:
            tft_vol_pred = tft_vol_walkforward(train_df, test_df, bank)
            tft_vol_pred = tft_vol_pred[:len(test_ret)]
            tft_vol_mae = mean_absolute_error(np.abs(test_ret), tft_vol_pred)
            results["tft_vol_pred"] = tft_vol_pred
            print(f"  TFT:       MAE = {tft_vol_mae:.6f}")
        except Exception as e:
            print(f"  TFT:       ERROR - {e}")
            tft_vol_mae = np.nan

        # Hybrid
        try:
            hybrid_vol_pred = hybrid_vol_walkforward(train_df, test_df, train_ret, test_ret, 0.5)
            hybrid_vol_mae = mean_absolute_error(np.abs(test_ret), hybrid_vol_pred)
            results["hybrid_vol_pred"] = hybrid_vol_pred
            print(f"  Hybrid:    MAE = {hybrid_vol_mae:.6f}")
        except Exception as e:
            print(f"  Hybrid:    ERROR - {e}")
            hybrid_vol_mae = np.nan

        # ===== PRICE PREDICTIONS =====
        print("\n--- PRICE ---")

        # Naive
        naive_price_pred = naive_price_walkforward(test_df)
        naive_price_mae = mean_absolute_error(test_df["close"].values, naive_price_pred)
        results["naive_price_pred"] = naive_price_pred
        print(f"  Naive:     MAE = {naive_price_mae:.6f}")

        # XGBoost
        try:
            xgb_price_pred = xgboost_price_walkforward(train_df, test_df)
            xgb_price_mae = mean_absolute_error(test_df["close"].values, xgb_price_pred)
            results["xgb_price_pred"] = xgb_price_pred
            print(f"  XGBoost:   MAE = {xgb_price_mae:.6f}")
        except Exception as e:
            print(f"  XGBoost:   ERROR - {e}")
            xgb_price_mae = np.nan

        # NeuralProphet
        try:
            np_price_pred = np_price_walkforward(train_df, test_df)
            np_price_pred = np_price_pred[:len(test_df)]
            np_price_mae = mean_absolute_error(test_df["close"].values, np_price_pred)
            results["np_price_pred"] = np_price_pred
            print(f"  NP:        MAE = {np_price_mae:.6f}")
        except Exception as e:
            print(f"  NP:        ERROR - {e}")
            np_price_mae = np.nan

        # TFT
        try:
            tft_price_pred = tft_price_walkforward(train_df, test_df, bank)
            tft_price_pred = tft_price_pred[:len(test_df)]
            tft_price_mae = mean_absolute_error(test_df["close"].values, tft_price_pred)
            results["tft_price_pred"] = tft_price_pred
            print(f"  TFT:       MAE = {tft_price_mae:.6f}")
        except Exception as e:
            print(f"  TFT:       ERROR - {e}")
            tft_price_mae = np.nan

        # Hybrid
        try:
            hybrid_price_pred = hybrid_price_walkforward(train_df, test_df, train_ret, test_ret, 0.3)
            hybrid_price_mae = mean_absolute_error(test_df["close"].values, hybrid_price_pred)
            results["hybrid_price_pred"] = hybrid_price_pred
            print(f"  Hybrid:    MAE = {hybrid_price_mae:.6f}")
        except Exception as e:
            print(f"  Hybrid:    ERROR - {e}")
            hybrid_price_mae = np.nan

        # Save per-day CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_DIR / f"{bank}_perday_all.csv", index=False)
        print(f"\nSaved: {OUTPUT_DIR / f'{bank}_perday_all.csv'}")

        # Find best models
        vol_maes = {"Naive": naive_vol_mae, "XGBoost": xgb_vol_mae, "NP": np_vol_mae, "TFT": tft_vol_mae, "Hybrid": hybrid_vol_mae}
        price_maes = {"Naive": naive_price_mae, "XGBoost": xgb_price_mae, "NP": np_price_mae, "TFT": tft_price_mae, "Hybrid": hybrid_price_mae}

        best_vol = min((k for k in vol_maes if not np.isnan(vol_maes[k])), key=lambda k: vol_maes[k])
        best_price = min((k for k in price_maes if not np.isnan(price_maes[k])), key=lambda k: price_maes[k])

        all_results.append({
            "bank": bank,
            "n_days": len(test_df),
            "naive_vol_mae": naive_vol_mae,
            "xgb_vol_mae": xgb_vol_mae,
            "np_vol_mae": np_vol_mae,
            "tft_vol_mae": tft_vol_mae,
            "hybrid_vol_mae": hybrid_vol_mae,
            "naive_price_mae": naive_price_mae,
            "xgb_price_mae": xgb_price_mae,
            "np_price_mae": np_price_mae,
            "tft_price_mae": tft_price_mae,
            "hybrid_price_mae": hybrid_price_mae,
            "best_vol": best_vol,
            "best_price": best_price,
        })

        # Create charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Chart 1: Volatility prediction
        ax1 = axes[0, 0]
        ax1.plot(df_results["ds"], df_results["actual_volatility"], "k-", alpha=0.7, label="Actual", linewidth=0.8)
        ax1.plot(df_results["ds"], df_results["xgb_vol_pred"], "orange", alpha=0.6, label="XGBoost", linewidth=0.8)
        ax1.plot(df_results["ds"], df_results["hybrid_vol_pred"], "r-", alpha=0.6, label="Hybrid", linewidth=0.8)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("|Log Return|")
        ax1.set_title(f"{bank} - Volatility: XGBoost vs Hybrid vs Actual")
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.tick_params(axis="x", rotation=45)

        # Chart 2: Price prediction
        ax2 = axes[0, 1]
        ax2.plot(df_results["ds"], df_results["actual_price"], "k-", alpha=0.7, label="Actual", linewidth=0.8)
        ax2.plot(df_results["ds"], df_results["naive_price_pred"], "gray", alpha=0.6, label="Naive", linewidth=0.8)
        ax2.plot(df_results["ds"], df_results["hybrid_price_pred"], "r-", alpha=0.6, label="Hybrid", linewidth=0.8)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.set_title(f"{bank} - Price: Hybrid vs Naive vs Actual")
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.tick_params(axis="x", rotation=45)

        # Chart 3: Volatility MAE comparison
        ax3 = axes[1, 0]
        models = ["Naive", "XGB", "NP", "TFT", "Hybrid"]
        colors = ["gray", "orange", "purple", "brown", "red"]
        vol_mae_values = [naive_vol_mae, xgb_vol_mae, np_vol_mae if not np.isnan(np_vol_mae) else 0,
                         tft_vol_mae if not np.isnan(tft_vol_mae) else 0, hybrid_vol_mae]
        bars = ax3.bar(models, vol_mae_values, color=colors)
        ax3.set_ylabel("MAE")
        ax3.set_title(f"{bank} - Volatility MAE (Lower is Better)")
        for bar, mae in zip(bars, vol_mae_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                    f"{mae:.4f}", ha="center", va="bottom", fontsize=9)

        # Chart 4: Price MAE comparison
        ax4 = axes[1, 1]
        price_mae_values = [naive_price_mae, xgb_price_mae if not np.isnan(xgb_price_mae) else 0,
                           np_price_mae if not np.isnan(np_price_mae) else 0,
                           tft_price_mae if not np.isnan(tft_price_mae) else 0, hybrid_price_mae]
        bars = ax4.bar(models, price_mae_values, color=colors)
        ax4.set_ylabel("MAE")
        ax4.set_title(f"{bank} - Price MAE (Lower is Better)")
        for bar, mae in zip(bars, price_mae_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{mae:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{bank}_all_models_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {OUTPUT_DIR / f'{bank}_all_models_comparison.png'}")

    # Save summary
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(OUTPUT_DIR / "perday_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY: 5 MODELS PER-DAY ANALYSIS")
    print("=" * 70)
    print("\nVolatility Prediction:")
    print(f"{'Bank':<6} {'Naive':>8} {'XGBoost':>8} {'NP':>8} {'TFT':>8} {'Hybrid':>8} {'Best':>8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['bank']:<6} {r['naive_vol_mae']:>8.4f} {r['xgb_vol_mae']:>8.4f} {r['np_vol_mae']:>8.4f} {r['tft_vol_mae']:>8.4f} {r['hybrid_vol_mae']:>8.4f} {r['best_vol']:>8}")

    print("\nPrice Prediction:")
    print(f"{'Bank':<6} {'Naive':>8} {'XGBoost':>8} {'NP':>8} {'TFT':>8} {'Hybrid':>8} {'Best':>8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['bank']:<6} {r['naive_price_mae']:>8.4f} {r['xgb_price_mae']:>8.4f} {r['np_price_mae']:>8.4f} {r['tft_price_mae']:>8.4f} {r['hybrid_price_mae']:>8.4f} {r['best_price']:>8}")

    print(f"\n\nSaved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()