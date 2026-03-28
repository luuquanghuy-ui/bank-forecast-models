from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}
RESULTS_CSV = SCRIPT_DIR / "pilot_garch_xgboost_results.csv"
REPORT_MD = SCRIPT_DIR / "pilot_garch_xgboost_results.md"


def fit_garch_11(returns: np.ndarray) -> tuple[float, float, float, float]:
    mu = float(np.mean(returns))
    eps = returns - mu
    variance = max(float(np.var(eps)), 1e-8)

    def neg_log_likelihood(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return np.inf

        sigma2 = np.empty_like(eps)
        sigma2[0] = variance
        for t in range(1, len(eps)):
            sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]
            if sigma2[t] <= 0:
                return np.inf

        return 0.5 * float(np.sum(np.log(sigma2) + (eps**2) / sigma2))

    initial = np.array([variance * 0.05, 0.05, 0.90])
    bounds = [(1e-10, variance * 10), (1e-8, 0.40), (1e-8, 0.999)]
    constraints = [{"type": "ineq", "fun": lambda x: 0.999 - x[1] - x[2]}]

    result = minimize(
        neg_log_likelihood,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    if not result.success:
        omega, alpha, beta = initial
    else:
        omega, alpha, beta = result.x

    return float(mu), float(omega), float(alpha), float(beta)


def garch_sigma_series(returns: np.ndarray, params: tuple[float, float, float, float]) -> np.ndarray:
    mu, omega, alpha, beta = params
    eps = returns - mu
    variance = max(float(np.var(eps)), 1e-8)
    sigma2 = np.empty_like(eps)
    sigma2[0] = variance

    for t in range(1, len(eps)):
        sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]

    return np.sqrt(sigma2)


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.clip(y_true, 1e-8, None)
    y_pred = np.clip(y_pred, 1e-8, None)
    return float(np.mean((y_true / y_pred) - np.log(y_true / y_pred) - 1))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 1e-8, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "qlike": qlike_loss(y_true, y_pred),
    }


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


def run_bank_pilot(bank: str, path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(path, parse_dates=["date"])
    df = prepare_features(raw)

    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    params = fit_garch_11(train["log_return"].to_numpy())
    sigma_full = garch_sigma_series(df["log_return"].to_numpy(), params)
    df["garch_sigma"] = sigma_full
    df["garch_sigma_sq"] = df["garch_sigma"] ** 2
    df["garch_z"] = (df["log_return"] - params[0]) / df["garch_sigma"].replace(0, np.nan)
    df["garch_sigma_lag_1"] = df["garch_sigma"].shift(1)
    df["garch_z_lag_1"] = df["garch_z"].shift(1)
    df = df.dropna().reset_index(drop=True)

    split_date = test["date"].iloc[0]
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()

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

    garch_features = base_features + ["garch_sigma", "garch_sigma_sq", "garch_z", "garch_sigma_lag_1", "garch_z_lag_1"]

    y_train = train["target_abs_return_next"].to_numpy()
    y_test = test["target_abs_return_next"].to_numpy()

    model_base = GradientBoostingRegressor(
        random_state=42,
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
    )
    model_garch = GradientBoostingRegressor(
        random_state=42,
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
    )

    model_base.fit(train[base_features], y_train)
    model_garch.fit(train[garch_features], y_train)

    predictions = {
        "Naive": test["abs_return"].to_numpy(),
        "GARCH-only": np.sqrt(2 / np.pi) * test["garch_sigma"].to_numpy(),
        "Boosting": model_base.predict(test[base_features]),
        "Boosting+GARCH": model_garch.predict(test[garch_features]),
    }

    rows: list[dict[str, object]] = []
    for model_name, preds in predictions.items():
        metrics = evaluate(y_test, preds)
        rows.append(
            {
                "bank": bank,
                "model": model_name,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_end": str(train["date"].max().date()),
                "test_start": str(test["date"].min().date()),
                "target": "abs(log_return_t+1)",
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "qlike": metrics["qlike"],
            }
        )

    summary = {
        "bank": bank,
        "rows_total": len(df),
        "rows_train": len(train),
        "rows_test": len(test),
        "train_end": str(train["date"].max().date()),
        "test_start": str(test["date"].min().date()),
        "garch_mu": params[0],
        "garch_omega": params[1],
        "garch_alpha": params[2],
        "garch_beta": params[3],
        "garch_persistence": params[2] + params[3],
    }

    return pd.DataFrame(rows), summary


def build_report(results: pd.DataFrame, summaries: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("# Pilot Demo GARCH-Boosting Cho 3 NgÃ¢n HÃ ng")
    lines.append("")
    lines.append("## Má»¥c tiÃªu")
    lines.append("")
    lines.append(
        "Cháº¡y thá»­ end-to-end trÃªn 3 file BID, CTG, VCB Ä‘á»ƒ xem hÆ°á»›ng tÃ¡ch riÃªng tá»«ng ngÃ¢n hÃ ng cÃ³ táº¡o Ä‘Æ°á»£c tÃ­n hiá»‡u dá»± bÃ¡o há»£p lÃ½ cho target `|log_return_{t+1}|` hay khÃ´ng."
    )
    lines.append("")
    lines.append("LÆ°u Ã½: mÃ´i trÆ°á»ng hiá»‡n táº¡i chÆ°a cÃ³ `xgboost` vÃ  `arch`, nÃªn báº£n pilot nÃ y dÃ¹ng:")
    lines.append("")
    lines.append("- `GradientBoostingRegressor` lÃ m proxy cho XGBoost")
    lines.append("- GARCH(1,1) tá»± Æ°á»›c lÆ°á»£ng báº±ng `scipy`")
    lines.append("")
    lines.append("## Thiáº¿t láº­p pilot")
    lines.append("")
    lines.append("- target: `abs(log_return_{t+1})`")
    lines.append("- split: 80% train, 20% test theo thá»i gian cho tá»«ng ngÃ¢n hÃ ng")
    lines.append("- benchmark: `Naive`, `GARCH-only`, `Boosting`, `Boosting+GARCH`")
    lines.append("- metrics: `MAE`, `RMSE`, `QLIKE`")
    lines.append("")
    lines.append("## TÃ³m táº¯t dá»¯ liá»‡u vÃ  tham sá»‘ GARCH")
    lines.append("")
    lines.append("| Bank | Rows | Train | Test | Train end | Test start | Alpha | Beta | Alpha+Beta |")
    lines.append("|---|---:|---:|---:|---|---|---:|---:|---:|")
    for item in summaries:
        lines.append(
            f"| {item['bank']} | {item['rows_total']} | {item['rows_train']} | {item['rows_test']} | "
            f"{item['train_end']} | {item['test_start']} | {item['garch_alpha']:.4f} | "
            f"{item['garch_beta']:.4f} | {item['garch_persistence']:.4f} |"
        )
    lines.append("")
    lines.append("## Káº¿t quáº£")
    lines.append("")

    for bank in results["bank"].unique():
        bank_rows = results[results["bank"] == bank].sort_values("rmse")
        lines.append(f"### {bank}")
        lines.append("")
        lines.append("| Model | MAE | RMSE | QLIKE |")
        lines.append("|---|---:|---:|---:|")
        for _, row in bank_rows.iterrows():
            lines.append(
                f"| {row['model']} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['qlike']:.6f} |"
            )
        best = bank_rows.iloc[0]
        lines.append("")
        lines.append(
            f"Káº¿t luáº­n nhanh: á»Ÿ `{bank}`, mÃ´ hÃ¬nh tá»‘t nháº¥t theo RMSE trong pilot nÃ y lÃ  **{best['model']}**."
        )
        lines.append("")

    mean_table = (
        results.groupby("model")[["mae", "rmse", "qlike"]]
        .mean()
        .sort_values("rmse")
        .reset_index()
    )
    lines.append("## Trung bÃ¬nh 3 ngÃ¢n hÃ ng")
    lines.append("")
    lines.append("| Model | Mean MAE | Mean RMSE | Mean QLIKE |")
    lines.append("|---|---:|---:|---:|")
    for _, row in mean_table.iterrows():
        lines.append(f"| {row['model']} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['qlike']:.6f} |")
    lines.append("")

    winner = mean_table.iloc[0]["model"]
    lines.append("## Diá»…n giáº£i ngáº¯n")
    lines.append("")
    lines.append(
        f"- Theo RMSE trung bÃ¬nh, mÃ´ hÃ¬nh tá»‘t nháº¥t cá»§a pilot lÃ  **{winner}**."
    )
    lines.append(
        "- Náº¿u `Boosting+GARCH` tháº¯ng hoáº·c bÃ¡m sÃ¡t `Boosting`, Ä‘iá»u Ä‘Ã³ cho tháº¥y hÆ°á»›ng tÃ¡ch 3 ngÃ¢n hÃ ng rá»“i thÃªm tÃ­n hiá»‡u GARCH lÃ  kháº£ thi Ä‘á»ƒ lÃ m tiáº¿p báº£n Ä‘áº§y Ä‘á»§."
    )
    lines.append(
        "- Náº¿u `Naive` váº«n ráº¥t máº¡nh, Ä‘Ã³ khÃ´ng pháº£i tÃ­n hiá»‡u xáº¥u; nÃ³ chá»‰ nÃ³i ráº±ng cáº§n tune ká»¹ hÆ¡n vÃ  chuyá»ƒn sang walk-forward validation á»Ÿ bÆ°á»›c tiáº¿p theo."
    )
    lines.append("")
    lines.append("## BÆ°á»›c tiáº¿p theo")
    lines.append("")
    lines.append("1. Thay proxy boosting báº±ng `xgboost` tháº­t khi cÃ i Ä‘Æ°á»£c thÆ° viá»‡n.")
    lines.append("2. Thay split 80/20 báº±ng walk-forward validation.")
    lines.append("3. ThÃªm benchmark `EGARCH`, `GJR-GARCH` vÃ  kiá»ƒm Ä‘á»‹nh `Diebold-Mariano`.")
    lines.append("4. Giá»¯ nguyÃªn hÆ°á»›ng: `1 ngÃ¢n hÃ ng = 1 pipeline riÃªng`.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    all_results: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for bank, path in BANK_FILES.items():
        bank_results, summary = run_bank_pilot(bank, path)
        all_results.append(bank_results)
        summaries.append(summary)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    REPORT_MD.write_text(build_report(results, summaries), encoding="utf-8")

    print(f"Saved results to: {RESULTS_CSV}")
    print(f"Saved report to: {REPORT_MD}")
    print()
    print(results.sort_values(["bank", "rmse"]).to_string(index=False))


if __name__ == "__main__":
    main()


