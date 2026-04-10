"""
4-Fold Walk-Forward: ALL MODELS ON BOTH TARGETS

Train ALL models on BOTH targets for fair comparison:
- VOLATILITY: Naive(0), GARCH, Ridge, RF, Hybrid, NP_vol
- PRICE: Naive(last), Ridge_price, RF_price, NP_price, TFT_price

Note: GARCH cannot predict price (not designed for it)
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from arch import arch_model
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

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


# ===== VOLATILITY MODELS =====

def garch_walkforward(train_ret, test_ret):
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


def ridge_vol_walkforward(train_df, test_df):
    feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["volatility"].values
    X_test = test_df[feature_cols].values
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def rf_vol_walkforward(train_df, test_df):
    feature_cols = ["volume_lag1", "volatility_5d", "volatility_20d", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["volatility"].values
    X_test = test_df[feature_cols].values
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def hybrid_walkforward(train_df, test_df, train_ret, test_ret, garch_weight=0.5):
    garch_pred = garch_walkforward(train_ret, test_ret)
    ridge_pred = ridge_vol_walkforward(train_df, test_df)
    return garch_weight * garch_pred + (1 - garch_weight) * ridge_pred


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
    # Replace NaN with 0
    pred_values = np.nan_to_num(pred_values, nan=0.0)
    return pred_values


# ===== PRICE MODELS =====

def naive_price_walkforward(test_df):
    """Predict last known price"""
    prices = test_df["close"].values
    pred = np.zeros(len(prices))
    pred[1:] = prices[:-1]
    return pred


def ridge_price_walkforward(train_df, test_df):
    """Ridge for price"""
    feature_cols = ["volume_lag1", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5", "price_lag1", "price_ma5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["close"].values
    X_test = test_df[feature_cols].values
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return np.nan_to_num(pred, nan=test_df["close"].iloc[0])


def rf_price_walkforward(train_df, test_df):
    """RF for price"""
    feature_cols = ["volume_lag1", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5", "price_lag1", "price_ma5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["close"].values
    X_test = test_df[feature_cols].values
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return np.nan_to_num(pred, nan=test_df["close"].iloc[0])


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
    # Replace NaN with last known price
    pred_values = np.nan_to_num(pred_values, nan=test_df["close"].iloc[0])
    return pred_values


def hybrid_price_walkforward(train_df, test_df, train_ret, test_ret, garch_weight=0.3):
    """Hybrid for price: GARCH volatility + Ridge for price
    Since GARCH doesn't predict price, we use GARCH's volatility signal
    as a feature combined with Ridge's price prediction.
    Train: use actual volatility from training period
    Test: use GARCH-predicted volatility as feature
    """
    # First, get GARCH volatility predictions for test period
    garch_vol_test = garch_walkforward(train_ret, test_ret)
    # For training, use the actual volatility from train period
    train_vol = np.abs(train_ret)

    # Create training features with actual volatility
    feature_cols = ["volume_lag1", "rsi_lag1", "return_lag1", "return_lag2", "return_lag5", "price_lag1", "price_ma5"]
    X_train = train_df[feature_cols].values
    y_train = train_df["close"].values

    # Add actual volatility for training
    X_train_with_vol = np.column_stack([X_train, train_vol])

    # For test: use GARCH predicted volatility
    X_test = test_df[feature_cols].values
    X_test_with_vol = np.column_stack([X_test, garch_vol_test])

    model = Ridge(alpha=1.0)
    model.fit(X_train_with_vol, y_train)
    pred = model.predict(X_test_with_vol)
    return np.nan_to_num(pred, nan=test_df["close"].iloc[0])


def run_comparison():
    print("=" * 70)
    print("4-FOLD WALK-FORWARD: ALL MODELS ON BOTH TARGETS")
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
            naive_vol_pred = np.zeros(len(test_ret))
            naive_vol_mae = mean_absolute_error(np.abs(test_ret), naive_vol_pred)

            garch_vol_pred = garch_walkforward(train_ret, test_ret)
            garch_vol_mae = mean_absolute_error(np.abs(test_ret), garch_vol_pred)

            ridge_vol_pred = ridge_vol_walkforward(train_df, test_df)
            ridge_vol_mae = mean_absolute_error(np.abs(test_ret), ridge_vol_pred)

            rf_vol_pred = rf_vol_walkforward(train_df, test_df)
            rf_vol_mae = mean_absolute_error(np.abs(test_ret), rf_vol_pred)

            hybrid_vol_pred = hybrid_walkforward(train_df, test_df, train_ret, test_ret, 0.5)
            hybrid_vol_mae = mean_absolute_error(np.abs(test_ret), hybrid_vol_pred)

            try:
                np_vol_pred = np_vol_walkforward(train_df, test_df)
                np_vol_pred = np_vol_pred[:len(test_ret)]
                np_vol_mae = mean_absolute_error(np.abs(test_ret), np_vol_pred)
            except Exception as e:
                np_vol_mae = np.nan

            print(f"    VOL: Naive={naive_vol_mae:.4f}, GARCH={garch_vol_mae:.4f}, Ridge={ridge_vol_mae:.4f}, RF={rf_vol_mae:.4f}, Hybrid={hybrid_vol_mae:.4f}, NP_vol={np_vol_mae if not np.isnan(np_vol_mae) else 'ERR':.4f}")

            # ===== PRICE PREDICTION =====
            naive_price_pred = naive_price_walkforward(test_df)
            naive_price_mae = mean_absolute_error(test_df["close"].values, naive_price_pred)

            ridge_price_pred = ridge_price_walkforward(train_df, test_df)
            ridge_price_mae = mean_absolute_error(test_df["close"].values, ridge_price_pred)

            rf_price_pred = rf_price_walkforward(train_df, test_df)
            rf_price_mae = mean_absolute_error(test_df["close"].values, rf_price_pred)

            try:
                np_price_pred = np_price_walkforward(train_df, test_df)
                np_price_pred = np_price_pred[:len(test_df)]
                np_price_mae = mean_absolute_error(test_df["close"].values, np_price_pred)
            except Exception as e:
                np_price_mae = np.nan

            hybrid_price_pred = hybrid_price_walkforward(train_df, test_df, train_ret, test_ret, 0.3)
            hybrid_price_mae = mean_absolute_error(test_df["close"].values, hybrid_price_pred)

            print(f"    PRI: Naive={naive_price_mae:.4f}, Ridge={ridge_price_mae:.4f}, RF={rf_price_mae:.4f}, NP={np_price_mae if not np.isnan(np_price_mae) else 'ERR':.4f}, Hybrid_price={hybrid_price_mae:.4f}")

            fold_results.append({
                "fold": fold_idx + 1,
                "train_n": train_end,
                "test_n": len(test_df),
                # Volatility
                "naive_vol_mae": naive_vol_mae,
                "garch_vol_mae": garch_vol_mae,
                "ridge_vol_mae": ridge_vol_mae,
                "rf_vol_mae": rf_vol_mae,
                "hybrid_vol_mae": hybrid_vol_mae,
                "np_vol_mae": np_vol_mae,
                # Price
                "naive_price_mae": naive_price_mae,
                "ridge_price_mae": ridge_price_mae,
                "rf_price_mae": rf_price_mae,
                "np_price_mae": np_price_mae,
                "hybrid_price_mae": hybrid_price_mae,
            })

        if fold_results:
            df_folds = pd.DataFrame(fold_results)
            avg = df_folds.mean(numeric_only=True)

            print(f"\n  AVERAGE ({len(fold_results)} folds):")
            print(f"    VOLATILITY:")
            print(f"      Naive:  {avg['naive_vol_mae']:.6f}")
            print(f"      GARCH:  {avg['garch_vol_mae']:.6f}")
            print(f"      Ridge:  {avg['ridge_vol_mae']:.6f}")
            print(f"      RF:     {avg['rf_vol_mae']:.6f}")
            print(f"      Hybrid: {avg['hybrid_vol_mae']:.6f}")
            print(f"      NP_vol: {avg['np_vol_mae']:.6f}")
            print(f"    PRICE:")
            print(f"      Naive:    {avg['naive_price_mae']:.6f}")
            print(f"      Ridge:    {avg['ridge_price_mae']:.6f}")
            print(f"      RF:       {avg['rf_price_mae']:.6f}")
            print(f"      NP:       {avg['np_price_mae']:.6f}")
            print(f"      Hybrid:   {avg['hybrid_price_mae']:.6f}")

            df_folds.to_csv(OUTPUT_DIR / f"{bank}_4fold_all_targets.csv", index=False)

            # Find best volatility
            vol_models = {
                "Naive": avg["naive_vol_mae"],
                "GARCH": avg["garch_vol_mae"],
                "Ridge": avg["ridge_vol_mae"],
                "RF": avg["rf_vol_mae"],
                "Hybrid": avg["hybrid_vol_mae"],
            }
            best_vol = min(vol_models, key=vol_models.get)

            # Find best price
            price_models = {
                "Naive": avg["naive_price_mae"],
                "Ridge": avg["ridge_price_mae"],
                "RF": avg["rf_price_mae"],
                "NP": avg["np_price_mae"],
                "Hybrid": avg["hybrid_price_mae"],
            }
            best_price = min(price_models, key=price_models.get)

            all_results.append({
                "bank": bank,
                "n_folds": len(fold_results),
                # Volatility
                "avg_naive_vol": avg["naive_vol_mae"],
                "avg_garch_vol": avg["garch_vol_mae"],
                "avg_ridge_vol": avg["ridge_vol_mae"],
                "avg_rf_vol": avg["rf_vol_mae"],
                "avg_hybrid_vol": avg["hybrid_vol_mae"],
                "avg_np_vol": avg["np_vol_mae"],
                # Price
                "avg_naive_price": avg["naive_price_mae"],
                "avg_ridge_price": avg["ridge_price_mae"],
                "avg_rf_price": avg["rf_price_mae"],
                "avg_np_price": avg["np_price_mae"],
                "avg_hybrid_price": avg["hybrid_price_mae"],
                "best_vol": best_vol,
                "best_price": best_price,
            })

    # Save summary
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(OUTPUT_DIR / "4fold_all_targets_summary.csv", index=False)

    # Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    banks = df_summary["bank"].values
    x = np.arange(len(banks))
    width = 0.15

    # Volatility chart
    ax1 = axes[0]
    vol_models_names = ["Naive", "GARCH", "Ridge", "RF", "Hybrid"]
    vol_colors = ["gray", "blue", "green", "orange", "red"]
    vol_maes = {
        "Naive": df_summary["avg_naive_vol"].values,
        "GARCH": df_summary["avg_garch_vol"].values,
        "Ridge": df_summary["avg_ridge_vol"].values,
        "RF": df_summary["avg_rf_vol"].values,
        "Hybrid": df_summary["avg_hybrid_vol"].values,
    }

    for i, (model, color) in enumerate(zip(vol_models_names, vol_colors)):
        ax1.bar(x + i*width, vol_maes[model], width, label=model, color=color)
    ax1.set_ylabel("MAE")
    ax1.set_title("Volatility Prediction: All Models")
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(banks)
    ax1.legend()

    # Price chart
    ax2 = axes[1]
    price_models_names = ["Naive", "Ridge", "RF", "NP", "Hybrid"]
    price_colors = ["gray", "green", "orange", "purple", "red"]
    price_maes = {
        "Naive": df_summary["avg_naive_price"].values,
        "Ridge": df_summary["avg_ridge_price"].values,
        "RF": df_summary["avg_rf_price"].values,
        "NP": df_summary["avg_np_price"].values,
        "Hybrid": df_summary["avg_hybrid_price"].values,
    }

    for i, (model, color) in enumerate(zip(price_models_names, price_colors)):
        ax2.bar(x + i*width, price_maes[model], width, label=model, color=color)
    ax2.set_ylabel("MAE")
    ax2.set_title("Price Prediction: ML Models")
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(banks)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4fold_all_targets_comparison.png", dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("SUMMARY: ALL MODELS ON BOTH TARGETS")
    print("=" * 70)
    for _, r in df_summary.iterrows():
        print(f"\n{r['bank']}:")
        print(f"  VOLATILITY: Naive={r['avg_naive_vol']:.4f}, GARCH={r['avg_garch_vol']:.4f}, Ridge={r['avg_ridge_vol']:.4f}, RF={r['avg_rf_vol']:.4f}, Hybrid={r['avg_hybrid_vol']:.4f}, NP_vol={r['avg_np_vol']:.4f}")
        print(f"  PRICE: Naive={r['avg_naive_price']:.4f}, Ridge={r['avg_ridge_price']:.4f}, RF={r['avg_rf_price']:.4f}, NP={r['avg_np_price']:.4f}, Hybrid_price={r['avg_hybrid_price']:.4f}")
        print(f"  BEST VOLATILITY: {r['best_vol']}, BEST PRICE: {r['best_price']}")

    print(f"\n\nSaved to: {OUTPUT_DIR}/")
    return df_summary


if __name__ == "__main__":
    run_comparison()
