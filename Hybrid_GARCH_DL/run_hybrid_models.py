"""
Hybrid GARCH-DL Approaches:

Approach 1: NeuralProphet predicting LOG RETURNS (instead of prices)
- Target: log(S_t) - log(S_{t-1}) = log_return
- Standard practice in finance (stationarity, variance stabilization)
- Compare: NP-Return vs Naive (predict 0) vs GARCH-only

Approach 2: NeuralProphet with GARCH SIGMA as lagged input feature
- GARCH sigma captures volatility clustering
- Add garch_sigma as lagged regressor
- Compare: NP-with-GARCH vs NP-without-GARCH vs GARCH-only
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from arch import arch_model


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "hybrid_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}

RESULTS_CSV = SCRIPT_DIR / "hybrid_results.csv"
REPORT_MD = SCRIPT_DIR / "hybrid_report.md"

DATE_COL = "date"
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.005
N_LAGS = 15

LAGGED_REGRESSORS_BASE = [
    "volume", "rsi", "volatility_20d",
    "vnindex_close", "vn30_close", "usd_vnd", "interest_rate",
]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data_with_returns(path: Path) -> pd.DataFrame:
    """Load data and compute log returns."""
    df = pd.read_csv(path, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    df = df.rename(columns={DATE_COL: "ds"})

    # Log return as target (standard in finance)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["return"] = df["close"].pct_change()

    # Price for reference
    df["y_price"] = df["close"]
    df["y"] = df["log_return"]  # NeuralProphet expects "y"
    df["y_log_return"] = df["log_return"]

    return df.dropna().reset_index(drop=True)


def prepare_data_with_garch(path: Path) -> pd.DataFrame:
    """Load data and compute GARCH features."""
    df = prepare_data_with_returns(path)

    returns_scaled = df["log_return"].to_numpy() * 100.0

    # Fit GARCH on full series (for feature engineering - only using past data)
    model = arch_model(returns_scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    result = model.fit(disp="off", show_warning=False)

    mu = float(result.params.get("mu", result.params.get("Const", 0.0))) / 100.0
    omega = float(result.params["omega"])
    alpha = float(result.params["alpha[1]"])
    beta = float(result.params["beta[1]"])

    sigma2 = np.empty_like(returns_scaled)
    sigma2[0] = max(float(np.var(returns_scaled)), 1e-8)
    eps = returns_scaled - mu * 100.0

    for t in range(1, len(returns_scaled)):
        sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]

    sigma = np.sqrt(sigma2) / 100.0

    df["garch_sigma"] = sigma
    df["garch_sigma_lag1"] = df["garch_sigma"].shift(1)
    df["garch_z"] = (df["log_return"] - mu) / np.clip(sigma, 1e-8, None)
    df["garch_z_lag1"] = df["garch_z"].shift(1)

    return df.dropna().reset_index(drop=True)


def split_df(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


# =============================================================================
# APPROACH 1: NeuralProphet predicting LOG RETURNS
# =============================================================================

def build_np_return_model():
    model = NeuralProphet(
        n_lags=N_LAGS,
        n_forecasts=1,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        loss_func="MSE",
    )
    model = model.add_lagged_regressor(LAGGED_REGRESSORS_BASE)
    model.set_plotting_backend("matplotlib")
    return model


def run_np_return(train, val, test):
    """NeuralProphet predicting log returns."""
    cols = ["ds", "y"] + LAGGED_REGRESSORS_BASE
    train_np = train[cols].copy()
    val_np = val[cols].copy()
    test_np = test[cols].copy()

    # Keep price for evaluation
    train_price = train["y_price"].copy()
    val_price = val["y_price"].copy()
    test_price = test["y_price"].copy()
    train_ret = train["y_log_return"].copy()
    val_ret = val["y_log_return"].copy()
    test_ret = test["y_log_return"].copy()

    # Validation model
    set_seed(42)
    model_val = build_np_return_model()
    model_val.fit(train_np, freq="B", validation_df=val_np, progress="none", metrics=False)

    val_input = pd.concat([train_np.tail(N_LAGS), val_np], ignore_index=True)
    val_forecast = model_val.predict(val_input)
    val_forecast = val_forecast[val_forecast["ds"].isin(val["ds"])]
    val_pred_ret = val_forecast["yhat1"].values

    # Test model
    set_seed(42)
    model_test = build_np_return_model()
    train_val_np = pd.concat([train_np, val_np], ignore_index=True)
    model_test.fit(train_val_np, freq="B", progress="none", metrics=False)

    test_input = pd.concat([train_val_np.tail(N_LAGS), test_np], ignore_index=True)
    test_forecast = model_test.predict(test_input)
    test_forecast = test_forecast[test_forecast["ds"].isin(test["ds"])]
    test_pred_ret = test_forecast["yhat1"].values

    return {
        "val_pred_ret": val_pred_ret,
        "val_actual_ret": val_ret.values,
        "val_price": val_price.values,
        "test_pred_ret": test_pred_ret,
        "test_actual_ret": test_ret.values,
        "test_price": test_price.values,
    }, model_test


def compute_return_metrics(actual_ret, pred_ret, price, model_name):
    """Compute metrics in return space and price space."""
    # Return-based metrics
    mae_ret = mean_absolute_error(actual_ret, pred_ret)
    rmse_ret = np.sqrt(mean_squared_error(actual_ret, pred_ret))

    # Convert to price: S_{t+1} = S_t * exp(r_pred)
    last_price = price[:-1]
    next_price_actual = price[1:]
    next_price_pred = last_price * np.exp(pred_ret[:len(last_price)])

    # Align lengths
    n = min(len(next_price_actual), len(next_price_pred))
    mae_price = mean_absolute_error(next_price_actual[:n], next_price_pred[:n])
    rmse_price = np.sqrt(mean_squared_error(next_price_actual[:n], next_price_pred[:n]))

    return {
        "model": model_name,
        "mae_return": mae_ret,
        "rmse_return": rmse_ret,
        "mae_price": mae_price,
        "rmse_price": rmse_price,
    }


def run_naive_return(test_price, test_ret):
    """Naive for returns = 0 (E[r] ≈ 0 for efficient markets)."""
    pred_ret = np.zeros(len(test_ret))
    mae_ret = mean_absolute_error(test_ret, pred_ret)
    rmse_ret = np.sqrt(mean_squared_error(test_ret, pred_ret))

    # Price prediction: S_{t+1} = S_t * exp(0) = S_t
    last_price = test_price[:-1]
    next_price_actual = test_price[1:]
    next_price_pred = last_price  # exp(0) = 1

    n = min(len(next_price_actual), len(next_price_pred))
    mae_price = mean_absolute_error(next_price_actual[:n], next_price_pred[:n])
    rmse_price = np.sqrt(mean_squared_error(next_price_actual[:n], next_price_pred[:n]))

    return {
        "model": "Naive (r=0)",
        "mae_return": mae_ret,
        "rmse_return": rmse_ret,
        "mae_price": mae_price,
        "rmse_price": rmse_price,
    }


def run_garch_only(train, test):
    """GARCH-only: predict volatility as proxy for |return|."""
    returns = train["log_return"].values * 100.0
    test_returns = test["log_return"].values
    test_sigma = test["garch_sigma"].values

    # Fit GARCH on train
    model = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    result = model.fit(disp="off", show_warning=False)

    # E[|r|] = sqrt(2/pi) * sigma for normal
    pred_abs_return = np.sqrt(2 / np.pi) * test_sigma

    mae = mean_absolute_error(np.abs(test_returns), pred_abs_return)
    rmse = np.sqrt(mean_squared_error(np.abs(test_returns), pred_abs_return))

    return {
        "model": "GARCH-only",
        "mae_return": mae,
        "rmse_return": rmse,
        "mae_price": np.nan,
        "rmse_price": np.nan,
    }


# =============================================================================
# APPROACH 2: NeuralProphet WITH GARCH SIGMA as feature
# =============================================================================

def build_np_garch_model(add_garch=True):
    model = NeuralProphet(
        n_lags=N_LAGS,
        n_forecasts=1,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        loss_func="MSE",
    )
    regressors = list(LAGGED_REGRESSORS_BASE)
    if add_garch:
        regressors.append("garch_sigma_lag1")
    model = model.add_lagged_regressor(regressors)
    model.set_plotting_backend("matplotlib")
    return model


def run_np_price_with_garch(train, val, test):
    """NeuralProphet predicting price, WITH garch_sigma as additional feature."""
    # NeuralProphet expects "y" as target column
    train_with_y = train.copy()
    val_with_y = val.copy()
    test_with_y = test.copy()
    train_with_y["y"] = train_with_y["y_price"]
    val_with_y["y"] = val_with_y["y_price"]
    test_with_y["y"] = test_with_y["y_price"]

    cols_base = ["ds", "y"] + LAGGED_REGRESSORS_BASE + ["garch_sigma_lag1"]
    train_np = train_with_y[cols_base].copy()
    val_np = val_with_y[cols_base].copy()
    test_np = test_with_y[cols_base].copy()

    val_actual = val["y_price"].values
    test_actual = test["y_price"].values

    # Without GARCH
    set_seed(42)
    model_no_garch = build_np_garch_model(add_garch=False)
    cols_no_garch = ["ds", "y"] + LAGGED_REGRESSORS_BASE
    train_np_no = train_with_y[cols_no_garch].copy()
    val_np_no = val_with_y[cols_no_garch].copy()
    test_np_no = test_with_y[cols_no_garch].copy()

    model_no_garch.fit(train_np_no, freq="B", validation_df=val_np_no, progress="none", metrics=False)
    test_input_no = pd.concat([train_np_no.tail(N_LAGS), test_np_no], ignore_index=True)
    forecast_no = model_no_garch.predict(test_input_no, drop_missing=True)
    forecast_no = forecast_no[forecast_no["ds"].isin(test["ds"])].dropna()
    pred_no = forecast_no["yhat1"].values

    # With GARCH
    set_seed(42)
    model_garch = build_np_garch_model(add_garch=True)
    model_garch.fit(train_np, freq="B", validation_df=val_np, progress="none", metrics=False)
    test_input = pd.concat([train_np.tail(N_LAGS), test_np], ignore_index=True)
    forecast_garch = model_garch.predict(test_input, drop_missing=True)
    forecast_garch = forecast_garch[forecast_garch["ds"].isin(test["ds"])].dropna()
    pred_garch = forecast_garch["yhat1"].values

    # Metrics
    mae_no = mean_absolute_error(test_actual[:len(pred_no)], pred_no)
    rmse_no = np.sqrt(mean_squared_error(test_actual[:len(pred_no)], pred_no))
    mae_garch = mean_absolute_error(test_actual[:len(pred_garch)], pred_garch)
    rmse_garch = np.sqrt(mean_squared_error(test_actual[:len(pred_garch)], pred_garch))

    return {
        "mae_np_no_garch": mae_no,
        "rmse_np_no_garch": rmse_no,
        "mae_np_with_garch": mae_garch,
        "rmse_np_with_garch": rmse_garch,
        "pred_no_garch": pred_no,
        "pred_with_garch": pred_garch,
    }, model_garch


def run_naive_price(train_price, test_price):
    """Naive: predict tomorrow = today. Uses last train price for first test prediction."""
    # Create full series: last train price + test prices
    last_train_price = train_price.iloc[-1]
    all_prices = pd.concat([pd.Series([last_train_price]), test_price], ignore_index=True)
    pred = all_prices.shift(1).iloc[1:].values  # shift and skip first
    actual = test_price.values
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return {"mae": mae, "rmse": rmse, "pred": pred}


def save_plot(bank, test_dates, test_price, pred, title, filename):
    out_path = OUTPUT_DIR / filename
    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, test_price, label="Actual", linewidth=2)
    plt.plot(test_dates, pred, label="Predicted", linewidth=2, alpha=0.8)
    plt.title(f"{bank} - {title}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_rows = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*60}")
        print(f"Processing {bank}")
        print(f"{'='*60}")

        # Data with GARCH features
        df_garch = prepare_data_with_garch(path)
        train, val, test = split_df(df_garch)

        print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

        # ---- APPROACH 1: Log Return Prediction ----
        print(f"\n--- Approach 1: NeuralProphet predicting log returns ---")

        # Naive (return = 0)
        naive_metrics = run_naive_return(test["y_price"].values, test["y_log_return"].values)

        # GARCH-only
        garch_metrics = run_garch_only(train, test)

        # NP predicting returns
        np_return_results, np_return_model = run_np_return(train, val, test)

        np_return_metrics = compute_return_metrics(
            np_return_results["test_actual_ret"],
            np_return_results["test_pred_ret"],
            np_return_results["test_price"],
            "NP-Return"
        )

        print(f"  Naive (r=0):    MAE_ret={naive_metrics['mae_return']:.6f}, RMSE_ret={naive_metrics['rmse_return']:.6f}")
        print(f"  NP-Return:      MAE_ret={np_return_metrics['mae_return']:.6f}, RMSE_ret={np_return_metrics['rmse_return']:.6f}")
        print(f"  GARCH-only:     MAE={garch_metrics['mae_return']:.6f}, RMSE={garch_metrics['rmse_return']:.6f}")

        # Save return forecast plot
        test_dates = test["ds"].values
        test_ret = np_return_results["test_actual_ret"]
        test_price = np_return_results["test_price"]
        pred_ret = np_return_results["test_pred_ret"]

        # Convert predictions to price for visualization
        last_price = test_price[:-1]
        next_price_actual = test_price[1:]
        next_price_pred_np = last_price * np.exp(pred_ret[:len(last_price)])

        plt.figure(figsize=(10, 5))
        plt.plot(test_dates[1:], next_price_actual, label="Actual next-day price", linewidth=2)
        plt.plot(test_dates[1:], next_price_pred_np, label="NP-Return prediction", linewidth=2, alpha=0.8)
        plt.plot(test_dates, test_price, label="Actual price", linewidth=1.5, alpha=0.5)
        plt.title(f"{bank} - NP-Return: Price Forecast from Return Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{bank}_approach1_np_return_forecast.png", dpi=150)
        plt.close()

        # ---- APPROACH 2: NP with GARCH sigma as feature ----
        print(f"\n--- Approach 2: NeuralProphet with GARCH sigma feature ---")

        naive_price_metrics = run_naive_price(train["y_price"], test["y_price"])
        np_garch_results, np_garch_model = run_np_price_with_garch(train, val, test)

        print(f"  Naive (S_t):   MAE={naive_price_metrics['mae']:.4f}, RMSE={naive_price_metrics['rmse']:.4f}")
        print(f"  NP (no GARCH): MAE={np_garch_results['mae_np_no_garch']:.4f}, RMSE={np_garch_results['rmse_np_no_garch']:.4f}")
        print(f"  NP (w/ GARCH): MAE={np_garch_results['mae_np_with_garch']:.4f}, RMSE={np_garch_results['rmse_np_with_garch']:.4f}")

        # Save price forecast plot
        save_plot(bank, test["ds"].values, test["y_price"].values,
                  np_garch_results["pred_with_garch"],
                  "NP with GARCH-sigma: Price Forecast",
                  f"{bank}_approach2_np_with_garch_forecast.png")

        # Save comparison plot
        plt.figure(figsize=(10, 5))
        plt.plot(test["ds"].values, test["y_price"].values, label="Actual", linewidth=2)
        plt.plot(test["ds"].values, np_garch_results["pred_no_garch"], label="NP (no GARCH)", linewidth=1.5, alpha=0.8)
        plt.plot(test["ds"].values, np_garch_results["pred_with_garch"], label="NP (with GARCH)", linewidth=1.5, alpha=0.8)
        plt.title(f"{bank} - NP with/without GARCH sigma comparison")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{bank}_approach2_comparison.png", dpi=150)
        plt.close()

        # Save test forecasts
        test[["ds", "y_price", "garch_sigma"]].assign(
            np_no_garch_pred=np_garch_results["pred_no_garch"],
            np_with_garch_pred=np_garch_results["pred_with_garch"],
        ).to_csv(OUTPUT_DIR / f"{bank}_approach2_forecasts.csv", index=False, encoding="utf-8-sig")

        # Store results
        row = {
            "bank": bank,
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            # Approach 1
            "naive_mae_return": naive_metrics["mae_return"],
            "naive_rmse_return": naive_metrics["rmse_return"],
            "np_return_mae_return": np_return_metrics["mae_return"],
            "np_return_rmse_return": np_return_metrics["rmse_return"],
            "garch_mae_return": garch_metrics["mae_return"],
            "garch_rmse_return": garch_metrics["rmse_return"],
            # Approach 2
            "naive_mae_price": naive_price_metrics["mae"],
            "naive_rmse_price": naive_price_metrics["rmse"],
            "np_no_garch_mae": np_garch_results["mae_np_no_garch"],
            "np_no_garch_rmse": np_garch_results["rmse_np_no_garch"],
            "np_with_garch_mae": np_garch_results["mae_np_with_garch"],
            "np_with_garch_rmse": np_garch_results["rmse_np_with_garch"],
        }
        all_rows.append(row)

    # Save results
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")

    # Build report
    lines = ["# Hybrid GARCH-DL Models - Results", ""]
    lines.append("## Approach 1: NeuralProphet predicting LOG RETURNS")
    lines.append("")
    lines.append("Target: log(S_t) - log(S_{t-1}) = log_return")
    lines.append("Benchmark: Naive (predict r=0), GARCH-only")
    lines.append("")

    for bank in results_df["bank"].unique():
        row = results_df[results_df["bank"] == bank].iloc[0]
        lines.append(f"### {bank}")
        lines.append("")
        lines.append("| Model | MAE (return) | RMSE (return) |")
        lines.append("|---|---:|---:|")
        lines.append(f"| Naive (r=0) | {row['naive_mae_return']:.6f} | {row['naive_rmse_return']:.6f} |")
        lines.append(f"| NP-Return | {row['np_return_mae_return']:.6f} | {row['np_return_rmse_return']:.6f} |")
        lines.append(f"| GARCH-only | {row['garch_mae_return']:.6f} | {row['garch_rmse_return']:.6f} |")
        lines.append("")

    lines.append("## Approach 2: NeuralProphet WITH GARCH sigma as feature")
    lines.append("")
    lines.append("Target: close price")
    lines.append("Additional feature: garch_sigma_lag1 (GARCH volatility, lagged)")
    lines.append("")

    for bank in results_df["bank"].unique():
        row = results_df[results_df["bank"] == bank].iloc[0]
        lines.append(f"### {bank}")
        lines.append("")
        lines.append("| Model | MAE (price) | RMSE (price) |")
        lines.append("|---|---:|---:|")
        lines.append(f"| Naive (S_t) | {row['naive_mae_price']:.4f} | {row['naive_rmse_price']:.4f} |")
        lines.append(f"| NP (no GARCH) | {row['np_no_garch_mae']:.4f} | {row['np_no_garch_rmse']:.4f} |")
        lines.append(f"| NP (with GARCH) | {row['np_with_garch_mae']:.4f} | {row['np_with_garch_rmse']:.4f} |")
        lines.append("")

    # Summary comparison
    lines.append("## Summary: Does GARCH sigma improve NP?")
    lines.append("")
    lines.append("| Bank | NP no GARCH RMSE | NP with GARCH RMSE | Improvement |")
    lines.append("|---|---:|---:|---:|")
    for bank in results_df["bank"].unique():
        row = results_df[results_df["bank"] == bank].iloc[0]
        diff = row['np_no_garch_rmse'] - row['np_with_garch_rmse']
        pct = diff / row['np_no_garch_rmse'] * 100 if row['np_no_garch_rmse'] != 0 else 0
        sign = "+" if diff > 0 else ""
        lines.append(f"| {bank} | {row['np_no_garch_rmse']:.4f} | {row['np_with_garch_rmse']:.4f} | {sign}{pct:.1f}% |")

    lines.append("")
    lines.append(f"Output folder: `{OUTPUT_DIR.name}`")
    lines.append(f"Results CSV: `{RESULTS_CSV.name}`")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved: {RESULTS_CSV}")
    print(f"Saved: {REPORT_MD}")
    print("\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    main()
