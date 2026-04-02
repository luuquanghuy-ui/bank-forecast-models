"""
Simple Hybrid: GARCH + NeuralProphet ensemble via weighted averaging
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
from sklearn.linear_model import Ridge
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


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path: Path):
    """Load data exactly like original working script."""
    TARGET_COL = "close"
    DATE_COL = "date"
    REGRESSORS = ["volume", "rsi", "volatility_20d", "vnindex_close", "vn30_close", "usd_vnd", "interest_rate"]

    df = pd.read_csv(path, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    cols = [DATE_COL, TARGET_COL] + [c for c in REGRESSORS if c in df.columns]
    df = df[cols].copy()
    df = df.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
    df["log_return"] = np.log(df["y"] / df["y"].shift(1))
    return df.dropna().reset_index(drop=True)


def split_data(df):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def fit_garch(train_df):
    """Fit GARCH(1,1) and return sigma for test period."""
    ret = train_df["log_return"].values
    ret_scaled = ret * 100.0
    model = arch_model(ret_scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = model.fit(disp="off", show_warning=False)

    mu = float(res.params.get("mu", res.params.get("Const", 0.0))) / 100.0
    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])

    # Walk-forward: predict sigma for test period iteratively
    test_ret = train_df["log_return"].values  # use train returns for rolling sigma
    sigma2 = np.empty(len(test_ret))
    sigma2[0] = max(float(np.var(ret_scaled)), 1e-8)
    eps = ret_scaled - mu * 100.0
    for t in range(1, len(test_ret)):
        sigma2[t] = omega + alpha * (eps[t-1]**2) + beta * sigma2[t-1]
    sigma_train = np.sqrt(sigma2) / 100.0

    return {
        "mu": mu, "omega": omega, "alpha": alpha, "beta": beta,
        "sigma_train": sigma_train
    }


def predict_garch_sigma(params, returns):
    """Predict GARCH sigma for given returns."""
    ret_scaled = returns * 100.0
    mu, omega, alpha, beta = params["mu"], params["omega"], params["alpha"], params["beta"]
    n = len(ret_scaled)
    sigma2 = np.empty(n)
    sigma2[0] = max(float(np.var(ret_scaled)), 1e-8)
    eps = ret_scaled - mu * 100.0
    for t in range(1, n):
        sigma2[t] = omega + alpha * (eps[t-1]**2) + beta * sigma2[t-1]
    return np.sqrt(sigma2) / 100.0


def run_approach1_np_return(train, val, test):
    """
    NeuralProphet predicting log_return.
    NeuralProphet target 'y' = log_return (stationary, not close price).
    """
    regressors = ["volume", "rsi", "volatility_20d", "vnindex_close", "vn30_close", "usd_vnd", "interest_rate"]

    # NeuralProphet expects 'y' as target - rename log_return to y
    train_df = train[["ds", "log_return"] + regressors].rename(columns={"log_return": "y"})
    val_df = val[["ds", "log_return"] + regressors].rename(columns={"log_return": "y"})
    test_df = test[["ds", "log_return"] + regressors].rename(columns={"log_return": "y"})

    set_seed(42)
    model = NeuralProphet(n_lags=15, n_forecasts=1, yearly_seasonality=False, weekly_seasonality=False,
                          daily_seasonality=False, epochs=50, batch_size=32, learning_rate=0.005, loss_func="MSE")
    model = model.add_lagged_regressor(regressors)
    model.set_plotting_backend("matplotlib")
    model.fit(train_df, freq="B", validation_df=val_df, progress="none", metrics=False)

    test_input = pd.concat([train_df.tail(15), test_df], ignore_index=True)
    forecast = model.predict(test_input)
    forecast = forecast[forecast["ds"].isin(test["ds"])].dropna()

    pred_ret = forecast["yhat1"].values
    actual_ret = forecast["ds"].map(dict(zip(test["ds"], test["log_return"]))).values

    mae_np = mean_absolute_error(actual_ret, pred_ret)
    rmse_np = np.sqrt(mean_squared_error(actual_ret, pred_ret))

    mae_naive = mean_absolute_error(actual_ret, np.zeros(len(actual_ret)))
    rmse_naive = np.sqrt(mean_squared_error(actual_ret, np.zeros(len(actual_ret))))

    params = fit_garch(train)
    garch_sigma = predict_garch_sigma(params, test["log_return"].values)
    pred_garch = np.sqrt(2/np.pi) * garch_sigma
    mae_garch = mean_absolute_error(np.abs(actual_ret), pred_garch)
    rmse_garch = np.sqrt(mean_squared_error(np.abs(actual_ret), pred_garch))

    return {
        "np_mae": mae_np, "np_rmse": rmse_np,
        "naive_mae": mae_naive, "naive_rmse": rmse_naive,
        "garch_mae": mae_garch, "garch_rmse": rmse_garch,
    }


def run_approach2_garch_enhanced_np(train, val, test):
    """
    Approach 2: NP + GARCH ensemble via weighted averaging.
    NP predicts log_return. Ensemble combines NP and GARCH volatility predictions.
    """
    regressors = ["volume", "rsi", "volatility_20d", "vnindex_close", "vn30_close", "usd_vnd", "interest_rate"]

    # GARCH params
    params = fit_garch(train)

    # NP on returns
    train_df = train[["ds", "log_return"] + regressors].rename(columns={"log_return": "y"})
    val_df = val[["ds", "log_return"] + regressors].rename(columns={"log_return": "y"})

    set_seed(42)
    model = NeuralProphet(n_lags=15, n_forecasts=1, yearly_seasonality=False, weekly_seasonality=False,
                          daily_seasonality=False, epochs=50, batch_size=32, learning_rate=0.005, loss_func="MSE")
    model = model.add_lagged_regressor(regressors)
    model.set_plotting_backend("matplotlib")
    model.fit(train_df, freq="B", validation_df=val_df, progress="none", metrics=False)

    # Test prediction
    test_df = test[["ds", "log_return"] + regressors].rename(columns={"log_return": "y"})
    test_input = pd.concat([train_df.tail(15), test_df], ignore_index=True)
    forecast = model.predict(test_input)
    forecast = forecast[forecast["ds"].isin(test["ds"])].dropna()

    pred_np = forecast["yhat1"].values
    actual_ret = forecast["ds"].map(dict(zip(test["ds"], test["log_return"]))).values

    garch_sigma_test = predict_garch_sigma(params, test["log_return"].values)
    pred_garch = np.sqrt(2/np.pi) * garch_sigma_test

    # Validate ensemble weight
    val_test_input = pd.concat([train_df.tail(15), val_df], ignore_index=True)
    val_forecast = model.predict(val_test_input)
    val_forecast = val_forecast[val_forecast["ds"].isin(val["ds"])].dropna()
    val_actual = val_forecast["ds"].map(dict(zip(val["ds"], val["log_return"]))).values
    val_np_pred = val_forecast["yhat1"].values
    val_garch_sigma = predict_garch_sigma(params, val["log_return"].values)
    val_garch_pred = np.sqrt(2/np.pi) * val_garch_sigma

    best_w, best_mae = 0, float('inf')
    for w in np.arange(0, 1.05, 0.05):
        pred_ens = w * val_np_pred + (1-w) * val_garch_pred
        mae = mean_absolute_error(np.abs(val_actual), pred_ens)
        if mae < best_mae:
            best_mae = mae
            best_w = w

    pred_ensemble = best_w * pred_np + (1-best_w) * pred_garch

    mae_np = mean_absolute_error(np.abs(actual_ret), pred_np)
    rmse_np = np.sqrt(mean_squared_error(np.abs(actual_ret), pred_np))
    mae_garch = mean_absolute_error(np.abs(actual_ret), pred_garch)
    rmse_garch = np.sqrt(mean_squared_error(np.abs(actual_ret), pred_garch))
    mae_ens = mean_absolute_error(np.abs(actual_ret), pred_ensemble)
    rmse_ens = np.sqrt(mean_squared_error(np.abs(actual_ret), pred_ensemble))

    return {
        "np_mae": mae_np, "np_rmse": rmse_np,
        "garch_mae": mae_garch, "garch_rmse": rmse_garch,
        "ensemble_mae": mae_ens, "ensemble_rmse": rmse_ens,
        "best_weight": best_w,
    }


def main():
    results = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        df = load_data(path)
        train, val, test = split_data(df)
        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

        print("Approach 1: NP predicting log returns...")
        a1 = run_approach1_np_return(train, val, test)
        print(f"  NP:     MAE={a1['np_mae']:.6f}, RMSE={a1['np_rmse']:.6f}")
        print(f"  Naive:  MAE={a1['naive_mae']:.6f}, RMSE={a1['naive_rmse']:.6f}")
        print(f"  GARCH:  MAE={a1['garch_mae']:.6f}, RMSE={a1['garch_rmse']:.6f}")

        print("Approach 2: NP + GARCH ensemble...")
        a2 = run_approach2_garch_enhanced_np(train, val, test)
        print(f"  NP:       MAE={a2['np_mae']:.6f}, RMSE={a2['np_rmse']:.6f}")
        print(f"  GARCH:    MAE={a2['garch_mae']:.6f}, RMSE={a2['garch_rmse']:.6f}")
        print(f"  Ensemble: MAE={a2['ensemble_mae']:.6f}, RMSE={a2['ensemble_rmse']:.6f} (weight={a2['best_weight']:.2f})")

        row = {"bank": bank, "train": len(train), "val": len(val), "test": len(test)}
        row.update(a1)
        row.update(a2)
        results.append(row)

    # Save
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")

    # Report
    lines = ["# Hybrid GARCH-DL Models - Results\n"]
    lines.append("## Approach 1: NeuralProphet predicting LOG RETURNS\n")
    lines.append("Target: log_return | Baseline: Naive (predict 0)\n")
    lines.append("\n| Bank | NP MAE | NP RMSE | Naive MAE | Naive RMSE | GARCH MAE | GARCH RMSE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(f"| {r['bank']} | {r['np_mae']:.6f} | {r['np_rmse']:.6f} | "
                     f"{r['naive_mae']:.6f} | {r['naive_rmse']:.6f} | "
                     f"{r['garch_mae']:.6f} | {r['garch_rmse']:.6f} |")

    lines.append("\n## Approach 2: NP + GARCH Ensemble\n")
    lines.append("Weighted average: w*NP + (1-w)*GARCH, w selected on validation\n")
    lines.append("\n| Bank | NP MAE | NP RMSE | GARCH MAE | GARCH RMSE | Ensemble MAE | Ensemble RMSE | Best w |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(f"| {r['bank']} | {r['np_mae']:.6f} | {r['np_rmse']:.6f} | "
                     f"{r['garch_mae']:.6f} | {r['garch_rmse']:.6f} | "
                     f"{r['ensemble_mae']:.6f} | {r['ensemble_rmse']:.6f} | {r['best_weight']:.2f} |")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved: {RESULTS_CSV}")
    print(f"Saved: {REPORT_MD}")
    print("\n" + df_results.to_string(index=False))


if __name__ == "__main__":
    main()
