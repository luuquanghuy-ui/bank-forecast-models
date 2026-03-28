from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
VENDOR_DIR = BASE_DIR / ".vendor_real"
if VENDOR_DIR.exists():
    sys.path.append(str(VENDOR_DIR))

from arch import arch_model
from xgboost import XGBRegressor

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}
RESULTS_CSV = SCRIPT_DIR / "fair_garch_xgboost_walkforward_results.csv"
REPORT_MD = SCRIPT_DIR / "fair_garch_xgboost_walkforward_results.md"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["target_abs_return_next"] = df["log_return"].shift(-1).abs()
    df["abs_return"] = df["log_return"].abs()
    df["log_volume"] = np.log(df["volume"].clip(lower=1))
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["open_close_gap"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["close_ma20"] = df["close"] / df["ma20"].replace(0, np.nan)
    df["close_ma50"] = df["close"] / df["ma50"].replace(0, np.nan)
    df["ma20_ma50"] = df["ma20"] / df["ma50"].replace(0, np.nan)
    df["vnindex_return"] = np.log(df["vnindex_close"] / df["vnindex_close"].shift(1))
    df["vn30_return"] = np.log(df["vn30_close"] / df["vn30_close"].shift(1))
    df["usd_vnd_change"] = np.log(df["usd_vnd"] / df["usd_vnd"].shift(1))
    df["interest_rate_change"] = df["interest_rate"].diff()

    for lag in range(1, 6):
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)
        df[f"abs_return_lag_{lag}"] = df["abs_return"].shift(lag)

    for lag in range(1, 4):
        df[f"vol20_lag_{lag}"] = df["volatility_20d"].shift(lag)
        df[f"volume_lag_{lag}"] = df["log_volume"].shift(lag)

    return df.dropna().reset_index(drop=True)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 1e-8, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def compute_garch_features(df: pd.DataFrame, train_idx: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    returns_scaled = df["log_return"].to_numpy() * 100.0
    train_scaled = returns_scaled[train_idx]

    model = arch_model(train_scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    result = model.fit(disp="off")

    mu = float(result.params.get("mu", result.params.get("Const", 0.0)))
    omega = float(result.params["omega"])
    alpha = float(result.params["alpha[1]"])
    beta = float(result.params["beta[1]"])

    sigma2 = np.empty_like(returns_scaled)
    sigma2[0] = max(float(np.var(train_scaled)), 1e-8)
    eps = returns_scaled - mu

    for t in range(1, len(returns_scaled)):
        sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]

    sigma = np.sqrt(sigma2) / 100.0

    enriched = df.copy()
    enriched["garch_sigma"] = sigma
    enriched["garch_sigma_sq"] = sigma**2
    enriched["garch_z"] = (df["log_return"].to_numpy() - (mu / 100.0)) / np.clip(sigma, 1e-8, None)
    enriched["garch_sigma_lag_1"] = enriched["garch_sigma"].shift(1)
    enriched["garch_z_lag_1"] = enriched["garch_z"].shift(1)
    enriched = enriched.dropna().reset_index(drop=True)

    params = {
        "mu": mu / 100.0,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
    }

    return enriched, params


def fit_xgb_with_inner_validation(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> np.ndarray:
    val_size = max(50, int(len(x_train) * 0.2))
    val_size = min(val_size, len(x_train) - 50)
    if val_size <= 0:
        val_size = max(10, len(x_train) // 5)

    x_fit = x_train.iloc[:-val_size]
    y_fit = y_train.iloc[:-val_size]
    x_val = x_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="rmse",
    )

    model.fit(
        x_fit,
        y_fit,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    return np.clip(model.predict(x_test), 1e-8, None)


def run_bank(bank: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, parse_dates=["date"])
    df = prepare_features(raw)

    base_features = [
        "close",
        "volume",
        "log_return",
        "abs_return",
        "ma20",
        "ma50",
        "rsi",
        "volatility_20d",
        "vnindex_close",
        "vn30_close",
        "interest_rate",
        "usd_vnd",
        "log_volume",
        "high_low_range",
        "open_close_gap",
        "close_ma20",
        "close_ma50",
        "ma20_ma50",
        "vnindex_return",
        "vn30_return",
        "usd_vnd_change",
        "interest_rate_change",
    ] + [f"return_lag_{lag}" for lag in range(1, 6)] + [f"abs_return_lag_{lag}" for lag in range(1, 6)] + [
        f"vol20_lag_{lag}" for lag in range(1, 4)
    ] + [f"volume_lag_{lag}" for lag in range(1, 4)]

    tscv = TimeSeriesSplit(n_splits=4)
    metric_rows: list[dict[str, object]] = []
    garch_rows: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        enriched, params = compute_garch_features(df, train_idx)

        # Rebuild exact fold windows by date after lag/dropna adjustments.
        test_start = df.iloc[test_idx[0]]["date"]
        test_end = df.iloc[test_idx[-1]]["date"]
        train = enriched[enriched["date"] < test_start].copy()
        test = enriched[(enriched["date"] >= test_start) & (enriched["date"] <= test_end)].copy()

        y_train = train["target_abs_return_next"]
        y_test = test["target_abs_return_next"].to_numpy()

        garch_features = base_features + [
            "garch_sigma",
            "garch_sigma_sq",
            "garch_z",
            "garch_sigma_lag_1",
            "garch_z_lag_1",
        ]

        preds = {
            "Naive": test["abs_return"].to_numpy(),
            "GARCH-only": np.sqrt(2 / np.pi) * test["garch_sigma"].to_numpy(),
            "XGBoost": fit_xgb_with_inner_validation(train[base_features], y_train, test[base_features]),
            "GARCH-XGBoost": fit_xgb_with_inner_validation(train[garch_features], y_train, test[garch_features]),
        }

        for model_name, y_pred in preds.items():
            metrics = evaluate(y_test, y_pred)
            metric_rows.append(
                {
                    "bank": bank,
                    "fold": fold,
                    "model": model_name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_end": str(train["date"].max().date()),
                    "test_start": str(test["date"].min().date()),
                    "test_end": str(test["date"].max().date()),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                }
            )

        garch_rows.append(
            {
                "bank": bank,
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_end": str(train["date"].max().date()),
                "test_start": str(test["date"].min().date()),
                "alpha": params["alpha"],
                "beta": params["beta"],
                "persistence": params["persistence"],
            }
        )

    return pd.DataFrame(metric_rows), pd.DataFrame(garch_rows)


def build_report(results: pd.DataFrame, garch_params: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# VÃ²ng 2 CÃ´ng Báº±ng: GARCH-XGBoost Vá»›i `arch` VÃ  `xgboost` Tháº­t")
    lines.append("")
    lines.append("## Thiáº¿t láº­p")
    lines.append("")
    lines.append("- 3 ngÃ¢n hÃ ng tÃ¡ch riÃªng: `BID`, `CTG`, `VCB`")
    lines.append("- target: `|log_return_{t+1}|`")
    lines.append("- `arch` dÃ¹ng Ä‘á»ƒ fit `GARCH(1,1)` tháº­t")
    lines.append("- `xgboost` dÃ¹ng Ä‘á»ƒ fit `XGBoost` vÃ  `GARCH-XGBoost` tháº­t")
    lines.append("- Ä‘Ã¡nh giÃ¡ báº±ng `walk-forward` vá»›i 4 folds cho tá»«ng ngÃ¢n hÃ ng")
    lines.append("- metric chÃ­nh: `RMSE`, metric phá»¥: `MAE`")
    lines.append("")
    lines.append("## Tham sá»‘ GARCH theo fold")
    lines.append("")
    lines.append("| Bank | Fold | Train end | Test start | Alpha | Beta | Alpha+Beta |")
    lines.append("|---|---:|---|---|---:|---:|---:|")
    for _, row in garch_params.iterrows():
        lines.append(
            f"| {row['bank']} | {int(row['fold'])} | {row['train_end']} | {row['test_start']} | "
            f"{row['alpha']:.4f} | {row['beta']:.4f} | {row['persistence']:.4f} |"
        )
    lines.append("")

    bank_summary = (
        results.groupby(["bank", "model"])[["mae", "rmse"]]
        .mean()
        .reset_index()
        .sort_values(["bank", "rmse"])
    )
    lines.append("## Káº¿t quáº£ trung bÃ¬nh theo ngÃ¢n hÃ ng")
    lines.append("")

    for bank in bank_summary["bank"].unique():
        subset = bank_summary[bank_summary["bank"] == bank]
        lines.append(f"### {bank}")
        lines.append("")
        lines.append("| Model | Mean MAE | Mean RMSE |")
        lines.append("|---|---:|---:|")
        for _, row in subset.iterrows():
            lines.append(f"| {row['model']} | {row['mae']:.6f} | {row['rmse']:.6f} |")
        best = subset.iloc[0]
        lines.append("")
        lines.append(f"Káº¿t luáº­n nhanh: á»Ÿ `{bank}`, model tá»‘t nháº¥t theo RMSE lÃ  **{best['model']}**.")
        lines.append("")

    overall = results.groupby("model")[["mae", "rmse"]].mean().reset_index().sort_values("rmse")
    lines.append("## Trung bÃ¬nh toÃ n bá»™")
    lines.append("")
    lines.append("| Model | Mean MAE | Mean RMSE |")
    lines.append("|---|---:|---:|")
    for _, row in overall.iterrows():
        lines.append(f"| {row['model']} | {row['mae']:.6f} | {row['rmse']:.6f} |")
    lines.append("")

    winner = overall.iloc[0]["model"]
    lines.append("## Káº¿t luáº­n")
    lines.append("")
    lines.append(f"- Theo RMSE trung bÃ¬nh cá»§a vÃ²ng 2 cÃ´ng báº±ng, model tá»‘t nháº¥t lÃ  **{winner}**.")
    lines.append("- Náº¿u `GARCH-XGBoost` váº«n thua `GARCH-only`, Ä‘Ã³ lÃ  báº±ng chá»©ng ráº±ng hybrid chÆ°a táº¡o thÃªm giÃ¡ trá»‹ trÃªn dataset nÃ y á»Ÿ target hiá»‡n táº¡i.")
    lines.append("- Náº¿u `GARCH-XGBoost` tiáº¿n gáº§n hoáº·c tháº¯ng á»Ÿ má»™t sá»‘ ngÃ¢n hÃ ng, váº«n cÃ³ thá»ƒ giá»¯ nÃ³ nhÆ° pháº§n so sÃ¡nh thá»±c nghiá»‡m.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    result_frames: list[pd.DataFrame] = []
    garch_frames: list[pd.DataFrame] = []

    for bank, path in BANK_FILES.items():
        metrics, params = run_bank(bank, path)
        result_frames.append(metrics)
        garch_frames.append(params)

    results = pd.concat(result_frames, ignore_index=True)
    garch_params = pd.concat(garch_frames, ignore_index=True)

    results.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    REPORT_MD.write_text(build_report(results, garch_params), encoding="utf-8")

    print(f"Saved results to: {RESULTS_CSV}")
    print(f"Saved report to: {REPORT_MD}")
    print()
    print(results.groupby(['bank', 'model'])[['mae', 'rmse']].mean().reset_index().sort_values(['bank', 'rmse']).to_string(index=False))


if __name__ == "__main__":
    main()


