from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE
from pytorch_forecasting.models import TemporalFusionTransformer


warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "tft_main_v2_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}

RESULTS_CSV = SCRIPT_DIR / "tft_main_v2_results.csv"
REPORT_MD = SCRIPT_DIR / "tft_main_v2_report.md"

DATE_COL = "date"
TARGET_COL = "close"
MAX_ENCODER_LENGTH = 30
MAX_PREDICTION_LENGTH = 1
BATCH_SIZE = 32
MAX_EPOCHS = 80
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 32

CALENDAR_CATEGORICALS = ["day_of_week", "month"]
EXTERNAL_REALS = [
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
    seed_everything(seed, workers=True)


def prepare_dataframe(path: Path, bank: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    cols = [DATE_COL, TARGET_COL] + [col for col in EXTERNAL_REALS if col in df.columns]
    df = df[cols].copy().dropna().reset_index(drop=True)

    # Log-transform price to reduce scale and improve model stability
    df["y_raw"] = df[TARGET_COL].copy()
    df["y"] = np.log(df[TARGET_COL])

    df = df.rename(columns={DATE_COL: "ds"})
    df["series"] = bank
    df["time_idx"] = np.arange(len(df))
    df["day_of_week"] = df["ds"].dt.dayofweek.astype(str)
    df["month"] = df["ds"].dt.month.astype(str)
    return df


def split_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def build_dataset(data: pd.DataFrame) -> TimeSeriesDataSet:
    return TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="y",
        group_ids=["series"],
        min_encoder_length=MAX_ENCODER_LENGTH,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=MAX_PREDICTION_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["series"],
        time_varying_known_categoricals=CALENDAR_CATEGORICALS,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["y"] + EXTERNAL_REALS,
        target_normalizer=GroupNormalizer(groups=["series"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        randomize_length=False,
    )


def build_model(dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        output_size=1,
        loss=RMSE(),
        log_interval=-1,
        reduce_on_plateau_patience=5,
    )


def build_trainer(with_validation: bool) -> Trainer:
    callbacks = []
    if with_validation:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=7, min_delta=1e-5, mode="min"))

    return Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        devices=1,
        gradient_clip_val=0.1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=callbacks,
    )


def collect_predictions(model: TemporalFusionTransformer, dataloader, full_df: pd.DataFrame) -> pd.DataFrame:
    prediction = model.predict(
        dataloader,
        return_index=True,
        return_y=True,
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )

    index_df = prediction.index.copy()
    pred_values = prediction.output.detach().cpu().numpy().reshape(-1)
    actual_values = prediction.y[0].detach().cpu().numpy().reshape(-1)

    merged = index_df.assign(y=actual_values, yhat1=pred_values)
    merged = merged.merge(full_df[["time_idx", "ds", "y_raw"]], on="time_idx", how="left")

    # Transform back from log to original scale
    merged["y_original"] = np.exp(merged["y"])
    merged["yhat1_original"] = np.exp(merged["yhat1"])

    merged = merged[["ds", "time_idx", "y", "yhat1", "y_original", "yhat1_original"]].sort_values("ds").reset_index(drop=True)
    return merged


def compute_metrics_original_scale(eval_df: pd.DataFrame) -> dict[str, float]:
    err = eval_df["y_original"] - eval_df["yhat1_original"]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def run_tft_main(
    df: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], TemporalFusionTransformer]:
    set_seed(42)
    train_dataset = build_dataset(train)
    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        df.iloc[: len(train) + len(val)].copy(),
        min_prediction_idx=int(val["time_idx"].min()),
        stop_randomization=True,
        predict=False,
    )
    train_loader = train_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    val_model = build_model(train_dataset)
    val_trainer = build_trainer(with_validation=True)
    val_trainer.fit(val_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_eval = collect_predictions(val_model, val_loader, df.iloc[: len(train) + len(val)].copy())

    set_seed(42)
    train_val = df.iloc[: len(train) + len(val)].copy()
    train_val_dataset = build_dataset(train_val)
    test_dataset = TimeSeriesDataSet.from_dataset(
        train_val_dataset,
        df.copy(),
        min_prediction_idx=int(test["time_idx"].min()),
        stop_randomization=True,
        predict=False,
    )
    train_val_loader = train_val_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    test_model = build_model(train_val_dataset)
    test_trainer = build_trainer(with_validation=False)
    test_trainer.fit(test_model, train_dataloaders=train_val_loader)
    test_eval = collect_predictions(test_model, test_loader, df.copy())

    val_metrics = compute_metrics_original_scale(val_eval)
    test_metrics = compute_metrics_original_scale(test_eval)
    metrics = {
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
    }
    return metrics, val_eval, test_eval, test_model


def run_naive(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    full = pd.concat([train[["ds", "y_raw"]], val[["ds", "y_raw"]], test[["ds", "y_raw"]]], ignore_index=True).copy()
    full["yhat1"] = full["y_raw"].shift(1)
    val_eval = full[full["ds"].isin(val["ds"])][["ds", "y_raw", "yhat1"]].dropna().copy()
    val_eval = val_eval.rename(columns={"y_raw": "y_original", "yhat1": "yhat1_original"})
    test_eval = full[full["ds"].isin(test["ds"])][["ds", "y_raw", "yhat1"]].dropna().copy()
    test_eval = test_eval.rename(columns={"y_raw": "y_original", "yhat1": "yhat1_original"})

    val_metrics = compute_metrics_original_scale(val_eval)
    test_metrics = compute_metrics_original_scale(test_eval)
    metrics = {
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
    }
    return metrics, val_eval, test_eval


def save_line_plot(bank: str, forecast_df: pd.DataFrame) -> None:
    out_path = OUTPUT_DIR / f"{bank}_tft_v2_test_plot.png"
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_df["ds"], forecast_df["y_original"], label="Actual", linewidth=2)
    plt.plot(forecast_df["ds"], forecast_df["yhat1_original"], label="Predicted", linewidth=2)
    plt.title(f"{bank} - TFT v2 Test Forecast (log-transformed)")
    plt.xlabel("Date")
    plt.ylabel("Log Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_report(results: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Temporal Fusion Transformer v2 - Improved Model Results")
    lines.append("")
    lines.append("## Changes from v1 (legitimate improvements):")
    lines.append("- hidden_size: 8 → 32 (model capacity)")
    lines.append("- hidden_continuous_size: 8 → 32")
    lines.append("- attention_head_size: 1 → 4")
    lines.append("- learning_rate: 0.01 → 0.001")
    lines.append("- max_epochs: 15 → 80")
    lines.append("- early_stopping patience: 3 → 7")
    lines.append("- reduce_on_plateau_patience: 3 → 5")
    lines.append("- batch_size: 64 → 32 (smaller for more updates)")
    lines.append("- Target: log(close) instead of raw close (standard practice in finance)")
    lines.append("- Target normalizer: softplus instead of default (for log-scale target)")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Main model: `TFT v2`")
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
        df = prepare_dataframe(path, bank)
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

        main_metrics, _, test_eval, fitted_model = run_tft_main(df, train, val, test)
        save_line_plot(bank, test_eval)
        test_eval.to_csv(OUTPUT_DIR / f"{bank}_tft_v2_test_forecast.csv", index=False, encoding="utf-8-sig")

        row = dict(common_meta)
        row.update({"model": "TFT v2"})
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
