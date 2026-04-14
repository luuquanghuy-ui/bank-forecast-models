"""
4-Fold Walk-Forward: 5 Models for Thesis

Models:
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

# TFT imports
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from lightning.pytorch import Trainer

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = BASE_DIR / "four_fold_all_targets"
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
    # GARCH volatility prediction
    garch_pred = garch_walkforward(train_ret, test_ret)

    # Ridge for features
    feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["volatility"].values
    X_test = test_df[feature_cols].values

    # Add GARCH volatility as feature
    train_vol = np.abs(train_ret)
    X_train_with_vol = np.column_stack([X_train, train_vol])
    X_test_with_vol = np.column_stack([X_test, garch_pred])

    model = Ridge(alpha=1.0)
    model.fit(X_train_with_vol, y_train)
    ridge_pred = model.predict(X_test_with_vol)

    return garch_weight * garch_pred + (1 - garch_weight) * ridge_pred


def hybrid_price_walkforward(train_df, test_df, train_ret, test_ret, garch_weight=0.3):
    """Hybrid for price: GARCH volatility signal + Ridge"""
    # GARCH volatility for signal
    garch_vol_test = garch_walkforward(train_ret, test_ret)
    train_vol = np.abs(train_ret)

    # Ridge with GARCH volatility as feature
    feature_cols = ["volume_lag1", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5", "price_lag1", "price_ma5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["close"].values
    X_test = test_df[feature_cols].values

    # Add GARCH volatility signal as additional feature
    X_train_with_vol = np.column_stack([X_train, train_vol])
    X_test_with_vol = np.column_stack([X_test, garch_vol_test])

    model = Ridge(alpha=1.0)
    model.fit(X_train_with_vol, y_train)
    pred = model.predict(X_test_with_vol)
    return np.nan_to_num(pred, nan=test_df["close"].iloc[0])


# ===== MAIN COMPARISON =====

def run_comparison():
    print("=" * 70)
    print("4-FOLD WALK-FORWARD: 5 MODELS FOR THESIS")
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

        fold_configs = [
            (0.50, 0.65, 0.70),
            (0.65, 0.80, 0.85),
            (0.80, 0.90, 0.95),
            (0.90, 0.95, 1.00),
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

            if len(test_df) < 30:
                continue

            print(f"\n  Fold {fold_idx+1}: train={train_end}, val={val_end-train_end}, test={test_end-val_end}")

            # ===== VOLATILITY PREDICTION =====
            # Naive
            naive_vol_pred = naive_vol_walkforward(test_ret)
            naive_vol_mae = mean_absolute_error(np.abs(test_ret), naive_vol_pred)

            # XGBoost
            xgb_vol_pred = xgboost_vol_walkforward(train_df, test_df)
            xgb_vol_mae = mean_absolute_error(np.abs(test_ret), xgb_vol_pred)

            # NP
            try:
                np_vol_pred = np_vol_walkforward(train_df, test_df)
                np_vol_pred = np_vol_pred[:len(test_ret)]
                np_vol_mae = mean_absolute_error(np.abs(test_ret), np_vol_pred)
            except Exception as e:
                np_vol_mae = np.nan

            # TFT
            try:
                tft_vol_pred = tft_vol_walkforward(train_df, test_df, bank)
                tft_vol_pred = tft_vol_pred[:len(test_ret)]
                tft_vol_mae = mean_absolute_error(np.abs(test_ret), tft_vol_pred)
            except Exception as e:
                tft_vol_mae = np.nan

            # Hybrid
            hybrid_vol_pred = hybrid_vol_walkforward(train_df, test_df, train_ret, test_ret, 0.5)
            hybrid_vol_mae = mean_absolute_error(np.abs(test_ret), hybrid_vol_pred)

            np_str = f"{np_vol_mae:.4f}" if not np.isnan(np_vol_mae) else "ERR"
            tft_str = f"{tft_vol_mae:.4f}" if not np.isnan(tft_vol_mae) else "ERR"
            print(f"    VOL: Naive={naive_vol_mae:.4f}, XGB={xgb_vol_mae:.4f}, NP={np_str}, TFT={tft_str}, Hybrid={hybrid_vol_mae:.4f}")

            # ===== PRICE PREDICTION =====
            # Naive
            naive_price_pred = naive_price_walkforward(test_df)
            naive_price_mae = mean_absolute_error(test_df["close"].values, naive_price_pred)

            # XGBoost
            xgb_price_pred = xgboost_price_walkforward(train_df, test_df)
            xgb_price_mae = mean_absolute_error(test_df["close"].values, xgb_price_pred)

            # NP
            try:
                np_price_pred = np_price_walkforward(train_df, test_df)
                np_price_pred = np_price_pred[:len(test_df)]
                np_price_mae = mean_absolute_error(test_df["close"].values, np_price_pred)
            except Exception as e:
                np_price_mae = np.nan

            # TFT
            try:
                tft_price_pred = tft_price_walkforward(train_df, test_df, bank)
                tft_price_pred = tft_price_pred[:len(test_df)]
                tft_price_mae = mean_absolute_error(test_df["close"].values, tft_price_pred)
            except Exception as e:
                tft_price_mae = np.nan

            # Hybrid
            hybrid_price_pred = hybrid_price_walkforward(train_df, test_df, train_ret, test_ret, 0.3)
            hybrid_price_mae = mean_absolute_error(test_df["close"].values, hybrid_price_pred)

            np_p_str = f"{np_price_mae:.4f}" if not np.isnan(np_price_mae) else "ERR"
            tft_p_str = f"{tft_price_mae:.4f}" if not np.isnan(tft_price_mae) else "ERR"
            print(f"    PRI: Naive={naive_price_mae:.4f}, XGB={xgb_price_mae:.4f}, NP={np_p_str}, TFT={tft_p_str}, Hybrid={hybrid_price_mae:.4f}")

            fold_results.append({
                "fold": fold_idx + 1,
                "train_n": train_end,
                "test_n": len(test_df),
                # Volatility
                "naive_vol_mae": naive_vol_mae,
                "xgb_vol_mae": xgb_vol_mae,
                "np_vol_mae": np_vol_mae,
                "tft_vol_mae": tft_vol_mae,
                "hybrid_vol_mae": hybrid_vol_mae,
                # Price
                "naive_price_mae": naive_price_mae,
                "xgb_price_mae": xgb_price_mae,
                "np_price_mae": np_price_mae,
                "tft_price_mae": tft_price_mae,
                "hybrid_price_mae": hybrid_price_mae,
            })

        if fold_results:
            df_folds = pd.DataFrame(fold_results)
            avg = df_folds.mean(numeric_only=True)

            print(f"\n  AVERAGE ({len(fold_results)} folds):")
            print(f"    VOLATILITY:")
            print(f"      Naive:   {avg['naive_vol_mae']:.6f}")
            print(f"      XGBoost: {avg['xgb_vol_mae']:.6f}")
            print(f"      NP:      {avg['np_vol_mae']:.6f}")
            print(f"      TFT:     {avg['tft_vol_mae']:.6f}")
            print(f"      Hybrid:  {avg['hybrid_vol_mae']:.6f}")
            print(f"    PRICE:")
            print(f"      Naive:   {avg['naive_price_mae']:.6f}")
            print(f"      XGBoost: {avg['xgb_price_mae']:.6f}")
            print(f"      NP:      {avg['np_price_mae']:.6f}")
            print(f"      TFT:     {avg['tft_price_mae']:.6f}")
            print(f"      Hybrid:  {avg['hybrid_price_mae']:.6f}")

            df_folds.to_csv(OUTPUT_DIR / f"{bank}_4fold_5models.csv", index=False)

            # Find best volatility
            vol_models = {
                "Naive": avg["naive_vol_mae"],
                "XGBoost": avg["xgb_vol_mae"],
                "NP": avg["np_vol_mae"],
                "TFT": avg["tft_vol_mae"],
                "Hybrid": avg["hybrid_vol_mae"],
            }
            best_vol = min(vol_models, key=vol_models.get)

            # Find best price
            price_models = {
                "Naive": avg["naive_price_mae"],
                "XGBoost": avg["xgb_price_mae"],
                "NP": avg["np_price_mae"],
                "TFT": avg["tft_price_mae"],
                "Hybrid": avg["hybrid_price_mae"],
            }
            best_price = min(price_models, key=price_models.get)

            all_results.append({
                "bank": bank,
                "n_folds": len(fold_results),
                # Volatility
                "avg_naive_vol": avg["naive_vol_mae"],
                "avg_xgb_vol": avg["xgb_vol_mae"],
                "avg_np_vol": avg["np_vol_mae"],
                "avg_tft_vol": avg["tft_vol_mae"],
                "avg_hybrid_vol": avg["hybrid_vol_mae"],
                # Price
                "avg_naive_price": avg["naive_price_mae"],
                "avg_xgb_price": avg["xgb_price_mae"],
                "avg_np_price": avg["np_price_mae"],
                "avg_tft_price": avg["tft_price_mae"],
                "avg_hybrid_price": avg["hybrid_price_mae"],
                "best_vol": best_vol,
                "best_price": best_price,
            })

    # Save summary
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(OUTPUT_DIR / "4fold_5models_summary.csv", index=False)

    # Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    banks = df_summary["bank"].values
    x = np.arange(len(banks))
    width = 0.15

    # Volatility chart
    ax1 = axes[0]
    models = ["Naive", "XGB", "NP", "TFT", "Hybrid"]
    colors = ["gray", "orange", "purple", "brown", "red"]
    vol_maes = {
        "Naive": df_summary["avg_naive_vol"].values,
        "XGB": df_summary["avg_xgb_vol"].values,
        "NP": df_summary["avg_np_vol"].values,
        "TFT": df_summary["avg_tft_vol"].values,
        "Hybrid": df_summary["avg_hybrid_vol"].values,
    }

    for i, (model, color) in enumerate(zip(models, colors)):
        ax1.bar(x + i*width, vol_maes[model], width, label=model, color=color)
    ax1.set_ylabel("MAE")
    ax1.set_title("Volatility Prediction: 5 Models")
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(banks)
    ax1.legend()

    # Price chart
    ax2 = axes[1]
    price_maes = {
        "Naive": df_summary["avg_naive_price"].values,
        "XGB": df_summary["avg_xgb_price"].values,
        "NP": df_summary["avg_np_price"].values,
        "TFT": df_summary["avg_tft_price"].values,
        "Hybrid": df_summary["avg_hybrid_price"].values,
    }

    for i, (model, color) in enumerate(zip(models, colors)):
        ax2.bar(x + i*width, price_maes[model], width, label=model, color=color)
    ax2.set_ylabel("MAE")
    ax2.set_title("Price Prediction: 5 Models")
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(banks)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4fold_5models_comparison.png", dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("SUMMARY: 5 MODELS FOR THESIS")
    print("=" * 70)
    for _, r in df_summary.iterrows():
        print(f"\n{r['bank']}:")
        print(f"  VOLATILITY: Naive={r['avg_naive_vol']:.4f}, XGB={r['avg_xgb_vol']:.4f}, NP={r['avg_np_vol']:.4f}, TFT={r['avg_tft_vol']:.4f}, Hybrid={r['avg_hybrid_vol']:.4f}")
        print(f"  PRICE: Naive={r['avg_naive_price']:.4f}, XGB={r['avg_xgb_price']:.4f}, NP={r['avg_np_price']:.4f}, TFT={r['avg_tft_price']:.4f}, Hybrid={r['avg_hybrid_price']:.4f}")
        print(f"  BEST VOLATILITY: {r['best_vol']}, BEST PRICE: {r['best_price']}")

    print(f"\n\nSaved to: {OUTPUT_DIR}/")
    return df_summary


if __name__ == "__main__":
    run_comparison()
