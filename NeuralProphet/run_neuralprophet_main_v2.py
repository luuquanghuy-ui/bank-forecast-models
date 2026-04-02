from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from neuralprophet import NeuralProphet


warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "neuralprophet_main_v2_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}

RESULTS_CSV = SCRIPT_DIR / "neuralprophet_main_v2_results.csv"
REPORT_MD = SCRIPT_DIR / "neuralprophet_main_v2_report.md"

TARGET_COL = "close"
DATE_COL = "date"
BASE_N_LAGS = 20
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

LAGGED_REGRESSORS = [
    "volume",
    "rsi",
    "volatility_20d",
    "vnindex_close",
    "vn30_close",
    "usd_vnd",
    "interest_rate",
]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    cols = [DATE_COL, TARGET_COL] + [c for c in LAGGED_REGRESSORS if c in df.columns]
    df = df[cols].copy()

    df = df.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
    # Log-transform: standard practice in finance to stabilize variance
    df["y"] = np.log(df["y"])
    df["y_raw"] = np.exp(df["y"])

    return df.dropna().reset_index(drop=True)


def split_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def build_model() -> NeuralProphet:
    model = NeuralProphet(
        n_lags=BASE_N_LAGS,
        n_forecasts=1,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        loss_func="MSE",
    )
    model = model.add_lagged_regressor(LAGGED_REGRESSORS)
    model.set_plotting_backend("matplotlib")
    return model


def compute_metrics_original_scale(eval_df: pd.DataFrame) -> dict[str, float]:
    err = eval_df["y_raw"] - eval_df["yhat1_original"]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def run_neuralprophet_main(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame, NeuralProphet]:
    # Extract y_raw before passing to NeuralProphet (it doesn't accept extra columns)
    train_y_raw = train["y_raw"].copy()
    val_y_raw = val["y_raw"].copy()
    test_y_raw = test["y_raw"].copy()

    # Build date -> y_raw mapping for evaluation
    all_dates = pd.concat([train["ds"], val["ds"], test["ds"]], ignore_index=True)
    all_y_raw = pd.concat([train_y_raw, val_y_raw, test_y_raw], ignore_index=True)
    date_to_yraw = dict(zip(all_dates, all_y_raw))

    # Keep only NeuralProphet-valid columns
    cols = ["ds", "y"] + LAGGED_REGRESSORS
    train_np = train[cols].copy()
    val_np = val[cols].copy()
    test_np = test[cols].copy()

    set_seed(42)
    model_val = build_model()
    model_val.fit(train_np, freq="B", validation_df=val_np, progress="none", metrics=False)

    val_input = pd.concat([train_np.tail(BASE_N_LAGS), val_np], ignore_index=True)
    val_forecast = model_val.predict(val_input)
    val_eval = val_forecast[val_forecast["ds"].isin(val["ds"])][["ds", "y", "yhat1"]].dropna().copy()
    val_eval["y_raw"] = val_eval["ds"].map(date_to_yraw)
    val_eval["yhat1_original"] = np.exp(val_eval["yhat1"])

    set_seed(42)
    model_test = build_model()
    train_val_np = pd.concat([train_np, val_np], ignore_index=True)
    model_test.fit(train_val_np, freq="B", progress="none", metrics=False)

    test_input = pd.concat([train_val_np.tail(BASE_N_LAGS), test_np], ignore_index=True)
    test_forecast = model_test.predict(test_input)
    test_eval = test_forecast[test_forecast["ds"].isin(test["ds"])][["ds", "y", "yhat1"]].dropna().copy()
    test_eval["y_raw"] = test_eval["ds"].map(date_to_yraw)
    test_eval["yhat1_original"] = np.exp(test_eval["yhat1"])

    val_metrics = compute_metrics_original_scale(val_eval)
    test_metrics = compute_metrics_original_scale(test_eval)
    metrics = {
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
    }
    return metrics, val_eval, test_eval, test_forecast, model_test


def run_naive(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    full = pd.concat([train[["ds", "y_raw"]], val[["ds", "y_raw"]], test[["ds", "y_raw"]]], ignore_index=True).copy()
    full["yhat1"] = full["y_raw"].shift(1)
    val_eval = full[full["ds"].isin(val["ds"])][["ds", "y_raw", "yhat1"]].dropna().copy()
    val_eval = val_eval.rename(columns={"yhat1": "yhat1_original"})
    test_eval = full[full["ds"].isin(test["ds"])][["ds", "y_raw", "yhat1"]].dropna().copy()
    test_eval = test_eval.rename(columns={"yhat1": "yhat1_original"})

    val_metrics = compute_metrics_original_scale(val_eval)
    test_metrics = compute_metrics_original_scale(test_eval)
    metrics = {
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
    }
    return metrics, val_eval, test_eval


def save_matplotlib_object(obj, out_path: Path) -> None:
    if hasattr(obj, "savefig"):
        fig = obj
    else:
        fig = plt.gcf()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_line_plot(bank: str, forecast_df: pd.DataFrame) -> None:
    out_path = OUTPUT_DIR / f"{bank}_neuralprophet_v2_test_plot.png"
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_df["ds"], forecast_df["y_raw"], label="Actual", linewidth=2)
    plt.plot(forecast_df["ds"], forecast_df["yhat1_original"], label="Predicted", linewidth=2)
    plt.title(f"{bank} - NeuralProphet v2 Test Forecast (log-transformed target)")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_model_specific_outputs(bank: str, model: NeuralProphet, full_forecast: pd.DataFrame, test_eval: pd.DataFrame) -> None:
    test_eval.to_csv(OUTPUT_DIR / f"{bank}_neuralprophet_v2_test_forecast.csv", index=False, encoding="utf-8-sig")
    full_forecast.to_csv(OUTPUT_DIR / f"{bank}_neuralprophet_v2_full_forecast.csv", index=False, encoding="utf-8-sig")
    save_line_plot(bank, test_eval)

    forecast_fig = model.plot(full_forecast)
    save_matplotlib_object(forecast_fig, OUTPUT_DIR / f"{bank}_neuralprophet_v2_forecast_plot.png")

    components_fig = model.plot_components(full_forecast)
    save_matplotlib_object(components_fig, OUTPUT_DIR / f"{bank}_neuralprophet_v2_components_plot.png")

    parameters_fig = model.plot_parameters()
    save_matplotlib_object(parameters_fig, OUTPUT_DIR / f"{bank}_neuralprophet_v2_parameters_plot.png")


def build_report(results: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# NeuralProphet v2 - Improved Model Results")
    lines.append("")
    lines.append("## Changes from v1 (legitimate improvements):")
    lines.append("- learning_rate: 0.01 → 0.001 (standard for neural nets)")
    lines.append("- n_lags: 10 → 20 (more historical context for daily data)")
    lines.append("- epochs: 40 → 100 (sufficient convergence)")
    lines.append("- batch_size: 64 → 32 (more gradient updates)")
    lines.append("- weekly_seasonality: True → False (trading days have no true weekly pattern)")
    lines.append("- Target: log(close) instead of raw close (variance stabilization)")
    lines.append("- Loss function: explicit MSE (clearer objective)")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Main model: `NeuralProphet v2`")
    lines.append("- Benchmark: `Naive`")
    lines.append("- Target: `log(close)` (transform back for evaluation)")
    lines.append("- Split: `70% train`, `15% validation`, `15% test`")
    lines.append("")
    for bank in results["bank"].unique():
        subset = results[results["bank"] == bank].sort_values("test_rmse")
        meta = subset.iloc[0]
        lines.append(f"## {bank}")
        lines.append("")
        lines.append(f"- Rows: `{int(meta['rows_total'])}` | Train: `{int(meta['train_rows'])}` | Val: `{int(meta['val_rows'])}` | Test: `{int(meta['test_rows'])}`")
        lines.append("")
        lines.append("| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in subset.iterrows():
            lines.append(f"| {row['model']} | {row['val_mae']:.4f} | {row['val_rmse']:.4f} | {row['test_mae']:.4f} | {row['test_rmse']:.4f} |")
        lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- Summary CSV: `{RESULTS_CSV.name}`")
    lines.append(f"- Output folder: `{OUTPUT_DIR.name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    all_rows: list[dict[str, object]] = []

    for bank, path in BANK_FILES.items():
        df = prepare_dataframe(path)
        train, val, test = split_df(df)

        common_meta = {
            "bank": bank,
            "rows_total": len(df),
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "train_end": str(train["ds"].max().date()),
            "val_start": str(val["ds"].min().date()),
            "test_start": str(test["ds"].min().date()),
        }

        naive_metrics, _, test_eval = run_naive(train, val, test)
        row = dict(common_meta)
        row.update({"model": "Naive"})
        row.update(naive_metrics)
        all_rows.append(row)

        main_metrics, _, test_eval, full_forecast, fitted_model = run_neuralprophet_main(train, val, test)
        save_model_specific_outputs(bank, fitted_model, full_forecast, test_eval)
        row = dict(common_meta)
        row.update({"model": "NeuralProphet v2"})
        row.update(main_metrics)
        all_rows.append(row)

    results = pd.DataFrame(all_rows).sort_values(["bank", "test_rmse"]).reset_index(drop=True)
    results.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    REPORT_MD.write_text(build_report(results), encoding="utf-8")

    print(f"Saved results to: {RESULTS_CSV}")
    print(f"Saved report to: {REPORT_MD}")
    print("")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
