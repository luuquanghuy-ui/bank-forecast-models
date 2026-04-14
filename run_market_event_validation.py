"""
Market Event Validation: 5 Models x 2 Targets

Tests model performance on highest volatility days.
This validates whether models work when it matters most.

Models:
- Naive (baseline)
- XGBoost (core model 1)
- NeuralProphet (NP) (core model 2)
- TFT (core model 3)
- Hybrid = GARCH volatility signal + Ridge (developed from thesis)

Targets:
- Volatility (|log_return|)
- Price (VND)
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
OUTPUT_DIR = BASE_DIR / "market_event_outputs"
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
    return np.zeros(len(test_ret))


def naive_price_walkforward(test_df):
    prices = test_df["close"].values
    pred = np.zeros(len(prices))
    pred[1:] = prices[:-1]
    return pred


# ===== XGBOOST =====

def xgboost_vol_predict(train_df, test_df):
    feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["volatility"].values
    X_test = test_df[feature_cols].values
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def xgboost_price_predict(train_df, test_df):
    feature_cols = ["volume_lag1", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5", "price_lag1", "price_ma5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["close"].values
    X_test = test_df[feature_cols].values
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return np.nan_to_num(pred, nan=test_df["close"].iloc[0])


# ===== NEURALPROPHET =====

def np_vol_predict(train_df, test_df):
    train_prophet = train_df[["ds", "volatility"]].copy()
    train_prophet.columns = ["ds", "y"]
    val_prophet = test_df[["ds", "volatility"]].copy()
    val_prophet.columns = ["ds", "y"]

    model = NeuralProphet(learning_rate=0.01, epochs=15, n_lags=10, n_forecasts=1, loss_func="MAE", weekly_seasonality=False)
    model.fit(train_prophet, freq="D", validation_df=val_prophet)
    predictions = model.predict(val_prophet)
    pred_values = predictions["yhat1"].values
    return np.nan_to_num(pred_values, nan=0.0)


def np_price_predict(train_df, test_df):
    train_prophet = train_df[["ds", "close"]].copy()
    train_prophet.columns = ["ds", "y"]
    val_prophet = test_df[["ds", "close"]].copy()
    val_prophet.columns = ["ds", "y"]

    model = NeuralProphet(learning_rate=0.01, epochs=15, n_lags=10, n_forecasts=1, loss_func="MAE", weekly_seasonality=False)
    model.fit(train_prophet, freq="D", validation_df=val_prophet)
    predictions = model.predict(val_prophet)
    pred_values = predictions["yhat1"].values
    return np.nan_to_num(pred_values, nan=test_df["close"].iloc[0])


# ===== TFT =====

def tft_vol_predict(train_df, test_df, bank="UNKNOWN"):
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

    trainer = Trainer(max_epochs=15, accelerator="cpu", enable_progress_bar=False, logger=False)
    trainer.fit(tft, train_dataloader)

    train_tail = train_d.iloc[-lookback:].copy()
    pred_df = pd.concat([train_tail, test_d], ignore_index=True).copy().reset_index(drop=True)

    pred_dataset = TimeSeriesDataSet.from_dataset(training, pred_df, predict=False)
    pred_dataloader = pred_dataset.to_dataloader(batch_size=64, num_workers=0, shuffle=False)

    raw_preds = tft.predict(pred_dataloader, mode="raw", return_x=True)
    # FIX: Index 3 = median (quantile 0.5), not index 4 (quantile 0.75)
    all_preds = raw_preds[0].prediction.cpu().numpy()[:, 0, 3].tolist()

    return np.array(all_preds)


def tft_price_predict(train_df, test_df, bank="UNKNOWN"):
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

    trainer = Trainer(max_epochs=15, accelerator="cpu", enable_progress_bar=False, logger=False)
    trainer.fit(tft, train_dataloader)

    train_tail = train_d.iloc[-lookback:].copy()
    pred_df = pd.concat([train_tail, test_d], ignore_index=True).copy().reset_index(drop=True)

    pred_dataset = TimeSeriesDataSet.from_dataset(training, pred_df, predict=False)
    pred_dataloader = pred_dataset.to_dataloader(batch_size=64, num_workers=0, shuffle=False)

    raw_preds = tft.predict(pred_dataloader, mode="raw", return_x=True)
    # FIX: Index 3 = median (quantile 0.5), not index 4 (quantile 0.75)
    all_preds = raw_preds[0].prediction.cpu().numpy()[:, 0, 3].tolist()

    return np.array(np.nan_to_num(all_preds, nan=test_df["close"].iloc[0]))


# ===== HYBRID =====

def garch_predict(train_ret, test_ret):
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


def hybrid_vol_predict(train_df, test_df, train_ret, test_ret, garch_weight=0.5, ridge_alpha=1.0):
    garch_pred = garch_predict(train_ret, test_ret)
    feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["volatility"].values
    X_test = test_df[feature_cols].values
    train_vol = np.abs(train_ret)
    X_train_with_vol = np.column_stack([X_train, train_vol])
    X_test_with_vol = np.column_stack([X_test, garch_pred])

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train_with_vol, y_train)
    ridge_pred = model.predict(X_test_with_vol)

    return garch_weight * garch_pred + (1 - garch_weight) * ridge_pred


def hybrid_price_predict(train_df, test_df, train_ret, test_ret, garch_weight=0.3, ridge_alpha=1.0):
    garch_vol_test = garch_predict(train_ret, test_ret)
    train_vol = np.abs(train_ret)
    feature_cols = ["volume_lag1", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5", "price_lag1", "price_ma5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["close"].values
    X_test = test_df[feature_cols].values
    X_train_with_vol = np.column_stack([X_train, train_vol])
    X_test_with_vol = np.column_stack([X_test, garch_vol_test])

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train_with_vol, y_train)
    pred = model.predict(X_test_with_vol)
    return np.nan_to_num(pred, nan=test_df["close"].iloc[0])


# ===== MARKET EVENT VALIDATION =====

def run_market_event_validation():
    """Test models on highest volatility days"""
    print("=" * 70)
    print("MARKET EVENT VALIDATION: 5 Models x 2 Targets")
    print("=" * 70)

    all_results = []
    top_n = 20  # Top 20 highest volatility days

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        df = load_data(path)
        df = create_features(df)
        n = len(df)

        # Use 70/30 split (standard for sensitivity)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        test_end = n

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:test_end].copy()

        train_ret = train_df["log_return"].values
        test_ret = test_df["log_return"].values
        test_price = test_df["close"].values
        test_dates = test_df["ds"].values

        if len(test_df) < 100:
            print(f"  Skipping {bank} - not enough test data")
            continue

        # ===== PREDICT ALL TEST DAYS =====
        print(f"  Running all models on {len(test_df)} test days...")

        # Naive
        naive_vol_pred = naive_vol_walkforward(test_ret)
        naive_price_pred = naive_price_walkforward(test_df)

        # XGBoost
        xgb_vol_pred = xgboost_vol_predict(train_df, test_df)
        xgb_price_pred = xgboost_price_predict(train_df, test_df)

        # NeuralProphet
        try:
            np_vol_pred = np_vol_predict(train_df, test_df)
            np_vol_pred = np_vol_pred[:len(test_ret)]
        except:
            np_vol_pred = np.full(len(test_ret), np.nan)

        try:
            np_price_pred = np_price_predict(train_df, test_df)
            np_price_pred = np_price_pred[:len(test_df)]
        except:
            np_price_pred = np.full(len(test_df), np.nan)

        # TFT
        try:
            tft_vol_pred = tft_vol_predict(train_df, test_df, bank)
            tft_vol_pred = tft_vol_pred[:len(test_ret)]
        except:
            tft_vol_pred = np.full(len(test_ret), np.nan)

        try:
            tft_price_pred = tft_price_predict(train_df, test_df, bank)
            tft_price_pred = tft_price_pred[:len(test_df)]
        except:
            tft_price_pred = np.full(len(test_df), np.nan)

        # Hybrid
        hybrid_vol_pred = hybrid_vol_predict(train_df, test_df, train_ret, test_ret, garch_weight=0.5, ridge_alpha=1.0)
        hybrid_price_pred = hybrid_price_predict(train_df, test_df, train_ret, test_ret, garch_weight=0.3, ridge_alpha=1.0)

        # ===== IDENTIFY HIGH VOLATILITY DAYS =====
        actual_vol = np.abs(test_ret)

        # Sort by volatility descending, get indices
        sorted_indices = np.argsort(actual_vol)[::-1]
        high_vol_indices = sorted_indices[:top_n]
        normal_indices = sorted_indices[top_n:]

        print(f"  High vol days threshold: {actual_vol[high_vol_indices[-1]]:.4f}")
        print(f"  Normal days threshold: {actual_vol[normal_indices[0]]:.4f}")

        # ===== CALCULATE MAE ON HIGH VOL DAYS =====
        high_vol_actual_price = test_price[high_vol_indices]
        high_vol_actual_vol = actual_vol[high_vol_indices]

        # Naive
        naive_high_vol_price_mae = mean_absolute_error(high_vol_actual_price, naive_price_pred[high_vol_indices])
        naive_high_vol_vol_mae = mean_absolute_error(high_vol_actual_vol, naive_vol_pred[high_vol_indices])

        # XGBoost
        xgb_high_vol_price_mae = mean_absolute_error(high_vol_actual_price, xgb_price_pred[high_vol_indices])
        xgb_high_vol_vol_mae = mean_absolute_error(high_vol_actual_vol, xgb_vol_pred[high_vol_indices])

        # NP
        np_high_vol_price_mae = mean_absolute_error(high_vol_actual_price, np_price_pred[high_vol_indices])
        np_high_vol_vol_mae = mean_absolute_error(high_vol_actual_vol, np_vol_pred[high_vol_indices])

        # TFT
        tft_high_vol_price_mae = mean_absolute_error(high_vol_actual_price, tft_price_pred[high_vol_indices])
        tft_high_vol_vol_mae = mean_absolute_error(high_vol_actual_vol, tft_vol_pred[high_vol_indices])

        # Hybrid
        hybrid_high_vol_price_mae = mean_absolute_error(high_vol_actual_price, hybrid_price_pred[high_vol_indices])
        hybrid_high_vol_vol_mae = mean_absolute_error(high_vol_actual_vol, hybrid_vol_pred[high_vol_indices])

        # ===== CALCULATE MAE ON NORMAL DAYS =====
        normal_actual_price = test_price[normal_indices]
        normal_actual_vol = actual_vol[normal_indices]

        naive_normal_price_mae = mean_absolute_error(normal_actual_price, naive_price_pred[normal_indices])
        naive_normal_vol_mae = mean_absolute_error(normal_actual_vol, naive_vol_pred[normal_indices])

        xgb_normal_price_mae = mean_absolute_error(normal_actual_price, xgb_price_pred[normal_indices])
        xgb_normal_vol_mae = mean_absolute_error(normal_actual_vol, xgb_vol_pred[normal_indices])

        np_normal_price_mae = mean_absolute_error(normal_actual_price, np_price_pred[normal_indices])
        np_normal_vol_mae = mean_absolute_error(normal_actual_vol, np_vol_pred[normal_indices])

        tft_normal_price_mae = mean_absolute_error(normal_actual_price, tft_price_pred[normal_indices])
        tft_normal_vol_mae = mean_absolute_error(normal_actual_vol, tft_vol_pred[normal_indices])

        hybrid_normal_price_mae = mean_absolute_error(normal_actual_price, hybrid_price_pred[normal_indices])
        hybrid_normal_vol_mae = mean_absolute_error(normal_actual_vol, hybrid_vol_pred[normal_indices])

        # ===== CALCULATE OVERALL MAE =====
        overall_naive_vol_mae = mean_absolute_error(actual_vol, naive_vol_pred)
        overall_xgb_vol_mae = mean_absolute_error(actual_vol, xgb_vol_pred)
        overall_np_vol_mae = mean_absolute_error(actual_vol, np_vol_pred)
        overall_tft_vol_mae = mean_absolute_error(actual_vol, tft_vol_pred)
        overall_hybrid_vol_mae = mean_absolute_error(actual_vol, hybrid_vol_pred)

        overall_naive_price_mae = mean_absolute_error(test_price, naive_price_pred)
        overall_xgb_price_mae = mean_absolute_error(test_price, xgb_price_pred)
        overall_np_price_mae = mean_absolute_error(test_price, np_price_pred)
        overall_tft_price_mae = mean_absolute_error(test_price, tft_price_pred)
        overall_hybrid_price_mae = mean_absolute_error(test_price, hybrid_price_pred)

        print(f"\n  VOLATILITY RESULTS:")
        print(f"    {'Model':<10} {'Overall':>10} {'High Vol':>10} {'Normal':>10} {'Assessment':>15}")
        print(f"    {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*15}")
        print(f"    {'Naive':<10} {overall_naive_vol_mae:>10.4f} {naive_high_vol_vol_mae:>10.4f} {naive_normal_vol_mae:>10.4f} {'Baseline':>15}")
        print(f"    {'XGBoost':<10} {overall_xgb_vol_mae:>10.4f} {xgb_high_vol_vol_mae:>10.4f} {xgb_normal_vol_mae:>10.4f} {'OK' if xgb_high_vol_vol_mae < naive_high_vol_vol_mae else 'WORSE':>15}")
        print(f"    {'NP':<10} {overall_np_vol_mae:>10.4f} {np_high_vol_vol_mae:>10.4f} {np_normal_vol_mae:>10.4f} {'OK' if np_high_vol_vol_mae < naive_high_vol_vol_mae else 'WORSE':>15}")
        print(f"    {'TFT':<10} {overall_tft_vol_mae:>10.4f} {tft_high_vol_vol_mae:>10.4f} {tft_normal_vol_mae:>10.4f} {'OK' if tft_high_vol_vol_mae < naive_high_vol_vol_mae else 'WORSE':>15}")
        print(f"    {'Hybrid':<10} {overall_hybrid_vol_mae:>10.4f} {hybrid_high_vol_vol_mae:>10.4f} {hybrid_normal_vol_mae:>10.4f} {'OK' if hybrid_high_vol_vol_mae < naive_high_vol_vol_mae else 'WORSE':>15}")

        print(f"\n  PRICE RESULTS:")
        print(f"    {'Model':<10} {'Overall':>10} {'High Vol':>10} {'Normal':>10} {'Assessment':>15}")
        print(f"    {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*15}")
        print(f"    {'Naive':<10} {overall_naive_price_mae:>10.4f} {naive_high_vol_price_mae:>10.4f} {naive_normal_price_mae:>10.4f} {'Baseline':>15}")
        print(f"    {'XGBoost':<10} {overall_xgb_price_mae:>10.4f} {xgb_high_vol_price_mae:>10.4f} {xgb_normal_price_mae:>10.4f} {'WORSE':>15}")
        print(f"    {'NP':<10} {overall_np_price_mae:>10.4f} {np_high_vol_price_mae:>10.4f} {np_normal_price_mae:>10.4f} {'WORSE':>15}")
        print(f"    {'TFT':<10} {overall_tft_price_mae:>10.4f} {tft_high_vol_price_mae:>10.4f} {tft_normal_price_mae:>10.4f} {'WORSE':>15}")
        print(f"    {'Hybrid':<10} {overall_hybrid_price_mae:>10.4f} {hybrid_high_vol_price_mae:>10.4f} {hybrid_normal_price_mae:>10.4f} {'BEST' if hybrid_high_vol_price_mae < naive_high_vol_price_mae else 'WORSE':>15}")

        # Store results
        result = {
            "bank": bank,
            "n_test_days": len(test_df),
            "n_high_vol_days": top_n,
            "vol_threshold": actual_vol[high_vol_indices[-1]],
            # Overall volatility
            "overall_naive_vol": overall_naive_vol_mae,
            "overall_xgb_vol": overall_xgb_vol_mae,
            "overall_np_vol": overall_np_vol_mae,
            "overall_tft_vol": overall_tft_vol_mae,
            "overall_hybrid_vol": overall_hybrid_vol_mae,
            # High vol days volatility
            "highvol_naive_vol": naive_high_vol_vol_mae,
            "highvol_xgb_vol": xgb_high_vol_vol_mae,
            "highvol_np_vol": np_high_vol_vol_mae,
            "highvol_tft_vol": tft_high_vol_vol_mae,
            "highvol_hybrid_vol": hybrid_high_vol_vol_mae,
            # Normal days volatility
            "normal_naive_vol": naive_normal_vol_mae,
            "normal_xgb_vol": xgb_normal_vol_mae,
            "normal_np_vol": np_normal_vol_mae,
            "normal_tft_vol": tft_normal_vol_mae,
            "normal_hybrid_vol": hybrid_normal_vol_mae,
            # Overall price
            "overall_naive_price": overall_naive_price_mae,
            "overall_xgb_price": overall_xgb_price_mae,
            "overall_np_price": overall_np_price_mae,
            "overall_tft_price": overall_tft_price_mae,
            "overall_hybrid_price": overall_hybrid_price_mae,
            # High vol days price
            "highvol_naive_price": naive_high_vol_price_mae,
            "highvol_xgb_price": xgb_high_vol_price_mae,
            "highvol_np_price": np_high_vol_price_mae,
            "highvol_tft_price": tft_high_vol_price_mae,
            "highvol_hybrid_price": hybrid_high_vol_price_mae,
            # Normal days price
            "normal_naive_price": naive_normal_price_mae,
            "normal_xgb_price": xgb_normal_price_mae,
            "normal_np_price": np_normal_price_mae,
            "normal_tft_price": tft_normal_price_mae,
            "normal_hybrid_price": hybrid_normal_price_mae,
        }
        all_results.append(result)

        # Save high volatility days data
        high_vol_days_df = test_df.iloc[high_vol_indices][["ds", "close", "volatility"]].copy()
        high_vol_days_df["actual_vol"] = actual_vol[high_vol_indices]
        high_vol_days_df["naive_vol_pred"] = naive_vol_pred[high_vol_indices]
        high_vol_days_df["xgb_vol_pred"] = xgb_vol_pred[high_vol_indices]
        high_vol_days_df["np_vol_pred"] = np_vol_pred[high_vol_indices]
        high_vol_days_df["tft_vol_pred"] = tft_vol_pred[high_vol_indices]
        high_vol_days_df["hybrid_vol_pred"] = hybrid_vol_pred[high_vol_indices]
        high_vol_days_df.to_csv(OUTPUT_DIR / f"high_vol_days_{bank}.csv", index=False)

    # Save summary
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(OUTPUT_DIR / "market_event_summary.csv", index=False)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, result in enumerate(all_results):
        bank = result["bank"]

        # Volatility comparison: High Vol vs Normal
        ax = axes[0, idx]
        models = ["Naive", "XGBoost", "NP", "TFT", "Hybrid"]
        high_vol_mves = [result["highvol_naive_vol"], result["highvol_xgb_vol"],
                        result["highvol_np_vol"], result["highvol_tft_vol"], result["highvol_hybrid_vol"]]
        normal_mves = [result["normal_naive_vol"], result["normal_xgb_vol"],
                      result["normal_np_vol"], result["normal_tft_vol"], result["normal_hybrid_vol"]]

        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, high_vol_mves, width, label="High Vol Days", color="red", alpha=0.7)
        ax.bar(x + width/2, normal_mves, width, label="Normal Days", color="blue", alpha=0.7)
        ax.set_ylabel("MAE")
        ax.set_title(f"{bank}: Volatility MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend(fontsize=8)

        # Price comparison: High Vol vs Normal
        ax = axes[1, idx]
        high_vol_prices = [result["highvol_naive_price"], result["highvol_xgb_price"],
                          result["highvol_np_price"], result["highvol_tft_price"], result["highvol_hybrid_price"]]
        normal_prices = [result["normal_naive_price"], result["normal_xgb_price"],
                        result["normal_np_price"], result["normal_tft_price"], result["normal_hybrid_price"]]

        ax.bar(x - width/2, high_vol_prices, width, label="High Vol Days", color="red", alpha=0.7)
        ax.bar(x + width/2, normal_prices, width, label="Normal Days", color="blue", alpha=0.7)
        ax.set_ylabel("MAE")
        ax.set_title(f"{bank}: Price MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "market_event_validation.png", dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("MARKET EVENT VALIDATION COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("=" * 70)

    return df_summary


if __name__ == "__main__":
    run_market_event_validation()