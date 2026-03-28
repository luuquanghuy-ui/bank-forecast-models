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
OUTPUT_DIR = SCRIPT_DIR / "tft_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}

RESULTS_CSV = SCRIPT_DIR / "tft_results.csv"
REPORT_MD = SCRIPT_DIR / "tft_results.md"

DATE_COL = "date"
TARGET_COL = "close"
MAX_ENCODER_LENGTH = 30
MAX_PREDICTION_LENGTH = 1
BATCH_SIZE = 64
MAX_EPOCHS = 15
LEARNING_RATE = 0.01
HIDDEN_SIZE = 8
ATTENTION_HEAD_SIZE = 1
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 8

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
    df = df.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})
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


def build_dataset(data: pd.DataFrame, use_covariates: bool) -> TimeSeriesDataSet:
    unknown_reals = ["y"] + (EXTERNAL_REALS if use_covariates else [])
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
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(groups=["series"]),
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
        reduce_on_plateau_patience=3,
    )


def build_trainer(with_validation: bool) -> Trainer:
    callbacks = []
    if with_validation:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=3, min_delta=1e-4, mode="min"))

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
    merged = merged.merge(full_df[["time_idx", "ds"]], on="time_idx", how="left")
    merged = merged[["ds", "time_idx", "y", "yhat1"]].sort_values("ds").reset_index(drop=True)
    return merged


def compute_metrics(eval_df: pd.DataFrame) -> dict[str, float]:
    err = eval_df["y"] - eval_df["yhat1"]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def run_tft_model(
    df: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_covariates: bool,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    set_seed(42)
    train_dataset = build_dataset(train, use_covariates=use_covariates)
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
    train_val_dataset = build_dataset(train_val, use_covariates=use_covariates)
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

    val_metrics = compute_metrics(val_eval)
    test_metrics = compute_metrics(test_eval)
    metrics = {
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
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
    lines.append("# Ket Qua Chay Temporal Fusion Transformer End-to-End Cho 3 Ngan Hang")
    lines.append("")
    lines.append("## Thiet lap")
    lines.append("")
    lines.append("- Du lieu tach rieng cho `BID`, `CTG`, `VCB`")
    lines.append("- Muc tieu du bao: `close`")
    lines.append("- Chia mau: `70% train`, `15% validation`, `15% test` theo thoi gian")
    lines.append(f"- `max_encoder_length = {MAX_ENCODER_LENGTH}`")
    lines.append(f"- `max_prediction_length = {MAX_PREDICTION_LENGTH}`")
    lines.append(f"- `max_epochs = {MAX_EPOCHS}`")
    lines.append(f"- `learning_rate = {LEARNING_RATE}`")
    lines.append("- Benchmark: `Naive`, `TFT baseline`, `TFT + covariates`")
    lines.append("")

    for bank in results["bank"].unique():
        subset = results[results["bank"] == bank].sort_values("test_rmse")
        meta = subset.iloc[0]
        lines.append(f"## {bank}")
        lines.append("")
        lines.append(
            f"- Rows tong: `{int(meta['rows_total'])}` | Train: `{int(meta['train_rows'])}` | Val: `{int(meta['val_rows'])}` | Test: `{int(meta['test_rows'])}`"
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
        lines.append(f"Ket luan nhanh: o `{bank}`, model tot nhat theo **Test RMSE** la **{best['model']}**.")
        lines.append("")

    overall = results.groupby("model")[["test_mae", "test_rmse"]].mean().reset_index().sort_values("test_rmse")
    lines.append("## Trung Binh 3 Ngan Hang")
    lines.append("")
    lines.append("| Model | Mean Test MAE | Mean Test RMSE |")
    lines.append("|---|---:|---:|")
    for _, row in overall.iterrows():
        lines.append(f"| {row['model']} | {row['test_mae']:.4f} | {row['test_rmse']:.4f} |")
    lines.append("")
    lines.append(f"Mo hinh tot nhat theo Test RMSE trung binh la **{overall.iloc[0]['model']}**.")
    lines.append("")
    lines.append("## File dau ra")
    lines.append("")
    lines.append(f"- CSV ket qua tong: `{RESULTS_CSV.name}`")
    lines.append(f"- Anh forecast test va CSV forecast chi tiet nam trong thu muc: `{OUTPUT_DIR.name}`")
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

        model_runs = [
            ("Naive", lambda a, b, c, d: run_naive(b, c, d)),
            ("TFT baseline", lambda a, b, c, d: run_tft_model(a, b, c, d, use_covariates=False)),
            ("TFT + covariates", lambda a, b, c, d: run_tft_model(a, b, c, d, use_covariates=True)),
        ]

        for model_name, runner in model_runs:
            metrics, _, test_eval = runner(df, train, val, test)
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
    print("")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
