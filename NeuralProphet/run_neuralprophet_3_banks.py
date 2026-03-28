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
OUTPUT_DIR = SCRIPT_DIR / "neuralprophet_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}

RESULTS_CSV = SCRIPT_DIR / "neuralprophet_results.csv"
REPORT_MD = SCRIPT_DIR / "neuralprophet_results.md"

TARGET_COL = "close"
DATE_COL = "date"
BASE_N_LAGS = 10
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.01

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
    return df.dropna().reset_index(drop=True)


def split_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def build_model(with_regressors: bool) -> NeuralProphet:
    model = NeuralProphet(
        n_lags=BASE_N_LAGS,
        n_forecasts=1,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )
    if with_regressors:
        model = model.add_lagged_regressor(LAGGED_REGRESSORS)
    return model


def run_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    with_regressors: bool,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    cols = ["ds", "y"] + (LAGGED_REGRESSORS if with_regressors else [])
    train = train[cols].copy()
    val = val[cols].copy()
    test = test[cols].copy()

    # Validation run
    set_seed(42)
    model_val = build_model(with_regressors=with_regressors)
    model_val.fit(train, freq="B", validation_df=val, progress="none", metrics=False)

    val_input = pd.concat([train.tail(BASE_N_LAGS), val], ignore_index=True)
    val_forecast = model_val.predict(val_input)
    val_eval = val_forecast[val_forecast["ds"].isin(val["ds"])][["ds", "y", "yhat1"]].dropna().copy()

    # Final train on train+val, evaluate on held-out test
    set_seed(42)
    model_test = build_model(with_regressors=with_regressors)
    train_val = pd.concat([train, val], ignore_index=True)
    model_test.fit(train_val, freq="B", progress="none", metrics=False)

    test_input = pd.concat([train_val.tail(BASE_N_LAGS), test], ignore_index=True)
    test_forecast = model_test.predict(test_input)
    test_eval = test_forecast[test_forecast["ds"].isin(test["ds"])][["ds", "y", "yhat1"]].dropna().copy()

    metrics = {
        "val_mae": float(np.mean(np.abs(val_eval["y"] - val_eval["yhat1"]))),
        "val_rmse": float(np.sqrt(np.mean((val_eval["y"] - val_eval["yhat1"]) ** 2))),
        "test_mae": float(np.mean(np.abs(test_eval["y"] - test_eval["yhat1"]))),
        "test_rmse": float(np.sqrt(np.mean((test_eval["y"] - test_eval["yhat1"]) ** 2))),
    }

    return metrics, val_eval, test_eval


def run_naive(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    full = pd.concat([train[["ds", "y"]], val[["ds", "y"]], test[["ds", "y"]]], ignore_index=True).copy()
    full["yhat1"] = full["y"].shift(1)

    val_eval = full[full["ds"].isin(val["ds"])][["ds", "y", "yhat1"]].dropna().copy()
    test_eval = full[full["ds"].isin(test["ds"])][["ds", "y", "yhat1"]].dropna().copy()

    metrics = {
        "val_mae": float(np.mean(np.abs(val_eval["y"] - val_eval["yhat1"]))),
        "val_rmse": float(np.sqrt(np.mean((val_eval["y"] - val_eval["yhat1"]) ** 2))),
        "test_mae": float(np.mean(np.abs(test_eval["y"] - test_eval["yhat1"]))),
        "test_rmse": float(np.sqrt(np.mean((test_eval["y"] - test_eval["yhat1"]) ** 2))),
    }

    return metrics, val_eval, test_eval


def save_plot(bank: str, model_name: str, forecast_df: pd.DataFrame) -> Path:
    safe_name = model_name.lower().replace("+", "_plus_").replace(" ", "_")
    out_path = OUTPUT_DIR / f"{bank}_{safe_name}_test_plot.png"

    plt.figure(figsize=(10, 5))
    plt.plot(forecast_df["ds"], forecast_df["y"], label="Actual", linewidth=2)
    plt.plot(forecast_df["ds"], forecast_df["yhat1"], label="Predicted", linewidth=2)
    plt.title(f"{bank} - {model_name} Test Forecast")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def save_forecast_csv(bank: str, model_name: str, forecast_df: pd.DataFrame) -> Path:
    safe_name = model_name.lower().replace("+", "_plus_").replace(" ", "_")
    out_path = OUTPUT_DIR / f"{bank}_{safe_name}_test_forecast.csv"
    forecast_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def build_report(results: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Káº¿t Quáº£ Cháº¡y NeuralProphet End-to-End Cho 3 NgÃ¢n HÃ ng")
    lines.append("")
    lines.append("## Thiáº¿t láº­p")
    lines.append("")
    lines.append("- Dá»¯ liá»‡u tÃ¡ch riÃªng cho `BID`, `CTG`, `VCB`")
    lines.append("- Má»¥c tiÃªu dá»± bÃ¡o: `close`")
    lines.append("- Chia máº«u: `70% train`, `15% validation`, `15% test` theo thá»i gian")
    lines.append("- `n_lags = 10`")
    lines.append("- `epochs = 40`")
    lines.append("- Benchmark: `Naive`, `NeuralProphet baseline`, `NeuralProphet + lagged regressors`")
    lines.append("")

    for bank in results["bank"].unique():
        subset = results[results["bank"] == bank].sort_values("test_rmse")
        meta = subset.iloc[0]
        lines.append(f"## {bank}")
        lines.append("")
        lines.append(
            f"- Rows tá»•ng: `{int(meta['rows_total'])}` | Train: `{int(meta['train_rows'])}` | Val: `{int(meta['val_rows'])}` | Test: `{int(meta['test_rows'])}`"
        )
        lines.append(
            f"- Train end: `{meta['train_end']}` | Val start: `{meta['val_start']}` | Test start: `{meta['test_start']}`"
        )
        lines.append("")
        lines.append("| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['model']} | {row['val_mae']:.4f} | {row['val_rmse']:.4f} | {row['test_mae']:.4f} | {row['test_rmse']:.4f} |"
            )
        best = subset.iloc[0]
        lines.append("")
        lines.append(f"Káº¿t luáº­n nhanh: á»Ÿ `{bank}`, model tá»‘t nháº¥t theo **Test RMSE** lÃ  **{best['model']}**.")
        lines.append("")

    overall = results.groupby("model")[["test_mae", "test_rmse"]].mean().reset_index().sort_values("test_rmse")
    lines.append("## Trung BÃ¬nh 3 NgÃ¢n HÃ ng")
    lines.append("")
    lines.append("| Model | Mean Test MAE | Mean Test RMSE |")
    lines.append("|---|---:|---:|")
    for _, row in overall.iterrows():
        lines.append(f"| {row['model']} | {row['test_mae']:.4f} | {row['test_rmse']:.4f} |")
    lines.append("")
    lines.append(f"MÃ´ hÃ¬nh tá»‘t nháº¥t theo Test RMSE trung bÃ¬nh lÃ  **{overall.iloc[0]['model']}**.")
    lines.append("")
    lines.append("## File Ä‘áº§u ra")
    lines.append("")
    lines.append(f"- CSV káº¿t quáº£ tá»•ng: `{RESULTS_CSV.name}`")
    lines.append(f"- áº¢nh forecast test vÃ  CSV forecast chi tiáº¿t náº±m trong thÆ° má»¥c: `{OUTPUT_DIR.name}`")
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

        model_runs = [
            ("Naive", run_naive),
            ("NeuralProphet baseline", lambda a, b, c: run_model(a, b, c, with_regressors=False)),
            ("NeuralProphet + lagged", lambda a, b, c: run_model(a, b, c, with_regressors=True)),
        ]

        for model_name, runner in model_runs:
            metrics, _, test_eval = runner(train, val, test)
            save_forecast_csv(bank, model_name, test_eval)
            save_plot(bank, model_name, test_eval)
            row = dict(common_meta)
            row.update({"model": model_name})
            row.update(metrics)
            all_rows.append(row)

    results = pd.DataFrame(all_rows).sort_values(["bank", "test_rmse"]).reset_index(drop=True)
    results.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    REPORT_MD.write_text(build_report(results), encoding="utf-8")

    print(f"Saved results to: {RESULTS_CSV}")
    print(f"Saved report to: {REPORT_MD}")
    print(f"Saved plots and detailed forecasts to: {OUTPUT_DIR}")
    print()
    print(results[["bank", "model", "val_mae", "val_rmse", "test_mae", "test_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()


