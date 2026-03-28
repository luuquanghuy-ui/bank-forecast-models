from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path(r"D:\labs 2\DOANPTDLKD\banks_master_dataset_final - banks_master_dataset_final.csv")


def fit_garch_11(returns: np.ndarray) -> tuple[float, float, float, float]:
    mu = float(np.mean(returns))
    eps = returns - mu
    variance = float(np.var(eps))
    variance = max(variance, 1e-8)

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


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def format_metrics(title: str, metrics: dict[str, float]) -> str:
    return (
        f"{title:<22}"
        f" MAE={metrics['mae']:.6f}"
        f"  RMSE={metrics['rmse']:.6f}"
        f"  R2={metrics['r2']:.4f}"
    )


def run_task(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    baseline_pred: np.ndarray,
    base_features: list[str],
    garch_features: list[str],
) -> list[str]:
    outputs: list[str] = []
    y_train = train[target]
    y_test = test[target]

    models = [
        ("Baseline", None, baseline_pred),
        ("Ridge", Ridge(alpha=1.0), None),
        (
            "Boosting",
            GradientBoostingRegressor(
                random_state=42,
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
            ),
            None,
        ),
        (
            "Boosting+GARCH",
            GradientBoostingRegressor(
                random_state=42,
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
            ),
            None,
        ),
    ]

    for name, model, pred_override in models:
        if pred_override is not None:
            preds = pred_override
        else:
            features = garch_features if name == "Boosting+GARCH" else base_features
            model.fit(train[features], y_train)
            preds = model.predict(test[features])

        metrics = evaluate_regression(y_test, preds)
        outputs.append(format_metrics(name, metrics))

    return outputs


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    for ticker, sub in df.groupby("ticker"):
        df.loc[sub.index, "target_return_next"] = sub["log_return"].shift(-1)
        df.loc[sub.index, "target_abs_return_next"] = sub["log_return"].shift(-1).abs()
        df.loc[sub.index, "target_vol20_next"] = sub["volatility_20d"].shift(-1)

    unique_dates = np.sort(df["date"].unique())
    split_date = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.80)])

    sigma_parts: list[pd.DataFrame] = []
    garch_params: dict[str, tuple[float, float, float, float]] = {}
    for ticker, sub in df.groupby("ticker", sort=True):
        sub = sub.sort_values("date").copy()
        train_mask = sub["date"] < split_date
        params = fit_garch_11(sub.loc[train_mask, "log_return"].to_numpy())
        sub["garch_sigma"] = garch_sigma_series(sub["log_return"].to_numpy(), params)
        sigma_parts.append(sub)
        garch_params[ticker] = params

    df = pd.concat(sigma_parts, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    df = pd.get_dummies(df, columns=["ticker"], drop_first=False, dtype=int)
    df = df.dropna().reset_index(drop=True)

    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()

    base_features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vnindex_close",
        "vn30_close",
        "log_return",
        "volatility_20d",
        "ma20",
        "ma50",
        "rsi",
        "usd_vnd",
        "interest_rate",
        "ticker_BID",
        "ticker_CTG",
        "ticker_VCB",
    ]
    garch_features = base_features + ["garch_sigma"]

    print("DATASET OVERVIEW")
    print(f"Rows used after target generation: {len(df)}")
    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Train end date: {(split_date - pd.Timedelta(days=1)).date()}")
    print(f"Test start date: {split_date.date()}")
    print()

    print("GARCH(1,1) PARAMS FIT ON TRAIN WINDOW")
    for ticker, (mu, omega, alpha, beta) in garch_params.items():
        print(
            f"{ticker}: mu={mu:.6f}, omega={omega:.8f}, "
            f"alpha={alpha:.4f}, beta={beta:.4f}, alpha+beta={alpha + beta:.4f}"
        )
    print()

    print("TASK 1: PREDICT NEXT-DAY LOG RETURN")
    baseline_return = np.zeros(len(test))
    for line in run_task(
        train=train,
        test=test,
        target="target_return_next",
        baseline_pred=baseline_return,
        base_features=base_features,
        garch_features=garch_features,
    ):
        print(line)
    sign_acc = float(
        np.mean(np.sign(test["target_return_next"].to_numpy()) == np.sign(baseline_return))
    )
    print(f"Zero-return baseline sign accuracy: {sign_acc:.4f}")
    print()

    print("TASK 2: PREDICT NEXT-DAY ABSOLUTE RETURN")
    baseline_abs = test["log_return"].abs().to_numpy()
    for line in run_task(
        train=train,
        test=test,
        target="target_abs_return_next",
        baseline_pred=baseline_abs,
        base_features=base_features,
        garch_features=garch_features,
    ):
        print(line)
    print(
        format_metrics(
            "GARCH-only",
            evaluate_regression(test["target_abs_return_next"], np.sqrt(2 / np.pi) * test["garch_sigma"]),
        )
    )
    print()

    print("TASK 3: PREDICT NEXT-DAY 20D VOLATILITY")
    baseline_vol = test["volatility_20d"].to_numpy()
    for line in run_task(
        train=train,
        test=test,
        target="target_vol20_next",
        baseline_pred=baseline_vol,
        base_features=base_features,
        garch_features=garch_features,
    ):
        print(line)
    print(
        format_metrics(
            "GARCH-only",
            evaluate_regression(test["target_vol20_next"], test["garch_sigma"]),
        )
    )


if __name__ == "__main__":
    main()
