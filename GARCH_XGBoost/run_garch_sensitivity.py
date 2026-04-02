"""
GARCH-XGBoost: Sensitivity Analysis + GARCH Params with p-values + Diebold-Mariano Test

1. Sensitivity to train/test split ratio
2. GARCH parameters table with z-stat and p-values
3. Diebold-Mariano test: GARCH-only vs Naive
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from arch import arch_model
from scipy import stats


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}

OUTPUT_DIR = SCRIPT_DIR / "sensitivity_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"date": "ds", "close": "y"})
    df["log_return"] = np.log(df["y"] / df["y"].shift(1))
    df["abs_return"] = df["log_return"].abs()
    df["target_abs_next"] = df["log_return"].shift(-1).abs()
    return df.dropna().reset_index(drop=True)


def garch_walkforward(train_ret, test_ret):
    """GARCH(1,1) walk-forward prediction."""
    ret_scaled = train_ret * 100.0
    model = arch_model(ret_scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = model.fit(disp="off", show_warning=False)

    mu = float(res.params.get("mu", res.params.get("Const", 0.0))) / 100.0
    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])

    sigma2 = np.empty(len(test_ret))
    sigma2_last = max(float(np.var(ret_scaled)), 1e-8)
    eps_last = (test_ret[0] - mu) * 100.0

    for i in range(len(test_ret)):
        sigma2[i] = omega + alpha * (eps_last ** 2) + beta * sigma2_last
        eps_last = (test_ret[i] - mu) * 100.0
        sigma2_last = sigma2[i]

    return np.sqrt(sigma2) / 100.0, res


def diebold_mariano(y_true, pred1, pred2):
    """
    Diebold-Mariano test for predictive accuracy.
    H0: Both forecasts have equal predictive ability.
    """
    e1 = y_true - pred1
    e2 = y_true - pred2
    d = np.abs(e1) - np.abs(e2)

    n = len(d)
    if n < 10:
        return np.nan, np.nan

    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1) / n

    if d_var < 1e-10:
        return 0.0, 1.0

    stat = d_mean / np.sqrt(d_var)
    p_value = 2 * (1 - stats.norm.cdf(abs(stat)))
    return stat, p_value


def run_split_sensitivity(path: Path, bank: str, splits: list):
    """Sensitivity to different train/test splits."""
    df = prepare_data(path)
    results = []

    for split_ratio in splits:
        train_size = int(len(df) * split_ratio)
        train = df.iloc[:train_size].copy()
        test = df.iloc[train_size:].copy()

        train_ret = train["log_return"].values
        test_ret = test["log_return"].values
        test_abs = test["target_abs_next"].values

        # GARCH walk-forward
        garch_sigma, res = garch_walkforward(train_ret, test_ret)
        garch_pred = np.sqrt(2 / np.pi) * garch_sigma
        naive_pred = np.zeros(len(test_ret))

        mae_naive = mean_absolute_error(test_abs, naive_pred)
        mae_garch = mean_absolute_error(test_abs, garch_pred)
        improvement = (mae_naive - mae_garch) / mae_naive * 100

        # Get params with standard errors
        mu = float(res.params.get("mu", res.params.get("Const", 0.0))) / 100.0
        omega = float(res.params["omega"])
        alpha = float(res.params["alpha[1]"])
        beta = float(res.params["beta[1]"])
        persistence = alpha + beta

        # Get standard errors
        mu_key = "mu" if "mu" in res.std_err else "Const"
        mu_se = float(res.std_err[mu_key])
        omega_se = float(res.std_err["omega"])
        alpha_se = float(res.std_err["alpha[1]"])
        beta_se = float(res.std_err["beta[1]"])

        results.append({
            "bank": bank,
            "split_ratio": f"{int(split_ratio*100)}/{int((1-split_ratio)*100)}",
            "train_n": len(train), "test_n": len(test),
            "naive_mae": mae_naive,
            "garch_mae": mae_garch,
            "improvement_pct": improvement,
            "mu": mu, "mu_se": mu_se,
            "omega": omega, "omega_se": omega_se,
            "alpha": alpha, "alpha_se": alpha_se,
            "beta": beta, "beta_se": beta_se,
            "persistence": persistence,
        })

        print(f"  {bank} | Split {int(split_ratio*100)}/{int((1-split_ratio)*100)} | "
              f"GARCH MAE={mae_garch:.6f} | Naive MAE={mae_naive:.6f} | Imp={improvement:.1f}%")

    return results


def run_fold_analysis(path: Path, bank: str):
    """Walk-forward 4-fold analysis with DM tests."""
    df = prepare_data(path)
    tscv = TimeSeriesSplit(n_splits=4)

    fold_results = []
    dm_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        train_ret = train["log_return"].values
        test_ret = test["log_return"].values
        test_abs = test["target_abs_next"].values

        garch_sigma, res = garch_walkforward(train_ret, test_ret)
        garch_pred = np.sqrt(2 / np.pi) * garch_sigma
        naive_pred = np.zeros(len(test_ret))

        mae_garch = mean_absolute_error(test_abs, garch_pred)
        mae_naive = mean_absolute_error(test_abs, naive_pred)

        # Get params
        mu = float(res.params.get("mu", res.params.get("Const", 0.0))) / 100.0
        omega = float(res.params["omega"])
        alpha = float(res.params["alpha[1]"])
        beta = float(res.params["beta[1]"])
        mu_key = "mu" if "mu" in res.std_err else "Const"
        mu_se = float(res.std_err[mu_key])
        omega_se = float(res.std_err["omega"])
        alpha_se = float(res.std_err["alpha[1]"])
        beta_se = float(res.std_err["beta[1]"])

        # z-stats and p-values
        mu_z = mu / mu_se if mu_se > 1e-10 else 0
        omega_z = omega / omega_se if omega_se > 1e-10 else 0
        alpha_z = alpha / alpha_se if alpha_se > 1e-10 else 0
        beta_z = beta / beta_se if beta_se > 1e-10 else 0

        mu_p = 2 * (1 - stats.norm.cdf(abs(mu_z)))
        omega_p = 2 * (1 - stats.norm.cdf(abs(omega_z)))
        alpha_p = 2 * (1 - stats.norm.cdf(abs(alpha_z)))
        beta_p = 2 * (1 - stats.norm.cdf(abs(beta_z)))

        fold_results.append({
            "bank": bank, "fold": fold,
            "train_n": len(train), "test_n": len(test),
            "train_end": str(train["ds"].max().date()),
            "test_start": str(test["ds"].min().date()),
            "test_end": str(test["ds"].max().date()),
            "naive_mae": mae_naive, "garch_mae": mae_garch,
            "mu": mu, "mu_se": mu_se, "mu_z": mu_z, "mu_p": mu_p,
            "omega": omega, "omega_se": omega_se, "omega_z": omega_z, "omega_p": omega_p,
            "alpha": alpha, "alpha_se": alpha_se, "alpha_z": alpha_z, "alpha_p": alpha_p,
            "beta": beta, "beta_se": beta_se, "beta_z": beta_z, "beta_p": beta_p,
            "persistence": alpha + beta,
            "mu_sig": "***" if mu_p < 0.01 else ("**" if mu_p < 0.05 else ("*" if mu_p < 0.1 else "")),
            "omega_sig": "***" if omega_p < 0.01 else ("**" if omega_p < 0.05 else ("*" if omega_p < 0.1 else "")),
            "alpha_sig": "***" if alpha_p < 0.01 else ("**" if alpha_p < 0.05 else ("*" if alpha_p < 0.1 else "")),
            "beta_sig": "***" if beta_p < 0.01 else ("**" if beta_p < 0.05 else ("*" if beta_p < 0.1 else "")),
        })

        # DM test
        dm_stat, dm_pval = diebold_mariano(test_abs, naive_pred, garch_pred)
        dm_results.append({
            "bank": bank, "fold": fold,
            "comparison": "GARCH vs Naive",
            "dm_stat": dm_stat, "p_value": dm_pval,
            "significant_05": "Yes" if dm_pval < 0.05 else "No",
            "significant_10": "Yes" if dm_pval < 0.1 else "No",
        })

    return fold_results, dm_results


def main():
    print("=" * 70)
    print("GARCH SENSITIVITY ANALYSIS + GARCH PARAMS + DIEBOLD-MARIANO")
    print("=" * 70)

    # Part 1: Sensitivity to split ratio
    print("\n" + "=" * 70)
    print("PART 1: SENSITIVITY TO TRAIN/TEST SPLIT")
    print("=" * 70)

    splits = [0.60, 0.70, 0.80]
    all_split_results = []

    for bank, path in BANK_FILES.items():
        print(f"\n{bank}:")
        results = run_split_sensitivity(path, bank, splits)
        all_split_results.extend(results)

    df_split = pd.DataFrame(all_split_results)
    df_split.to_csv(OUTPUT_DIR / "sensitivity_split_ratios.csv", index=False, encoding="utf-8-sig")

    # Part 2: 4-fold analysis with GARCH params and DM test
    print("\n" + "=" * 70)
    print("PART 2: WALK-FORWARD 4-FOLD ANALYSIS")
    print("=" * 70)

    all_fold_results = []
    all_dm_results = []

    for bank, path in BANK_FILES.items():
        print(f"\n{bank}:")
        fold_results, dm_results = run_fold_analysis(path, bank)
        all_fold_results.extend(fold_results)
        all_dm_results.extend(dm_results)

        for fr in fold_results:
            print(f"  Fold {fr['fold']}: alpha={fr['alpha']:.4f}{fr['alpha_sig']}, "
                  f"beta={fr['beta']:.4f}{fr['beta_sig']}, "
                  f"alpha+beta={fr['persistence']:.4f}, "
                  f"GARCH MAE={fr['garch_mae']:.6f}")

    df_folds = pd.DataFrame(all_fold_results)
    df_dm = pd.DataFrame(all_dm_results)

    df_folds.to_csv(OUTPUT_DIR / "garch_params_4fold.csv", index=False, encoding="utf-8-sig")
    df_dm.to_csv(OUTPUT_DIR / "diebold_mariano_results.csv", index=False, encoding="utf-8-sig")

    # Print summary tables
    print("\n" + "=" * 70)
    print("SUMMARY: GARCH PARAMETERS WITH SIGNIFICANCE")
    print("=" * 70)

    for bank in ["BID", "CTG", "VCB"]:
        print(f"\n{bank}:")
        rows = df_folds[df_folds["bank"] == bank]
        for _, r in rows.iterrows():
            print(f"  Fold {r['fold']} ({r['train_end']} to {r['test_end']}):")
            print(f"    mu    = {r['mu']:.6f} (z={r['mu_z']:.2f}, p={r['mu_p']:.4f}){r['mu_sig']}")
            print(f"    omega = {r['omega']:.4f} (z={r['omega_z']:.2f}, p={r['omega_p']:.4f}){r['omega_sig']}")
            print(f"    alpha = {r['alpha']:.4f} (z={r['alpha_z']:.2f}, p={r['alpha_p']:.4f}){r['alpha_sig']}")
            print(f"    beta  = {r['beta']:.4f} (z={r['beta_z']:.2f}, p={r['beta_p']:.4f}){r['beta_sig']}")
            print(f"    alpha+beta = {r['persistence']:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY: DIEBOLD-MARIANO TEST RESULTS")
    print("=" * 70)
    print("\nGARCH vs Naive (H0: equal predictive ability):")
    for _, r in df_dm.iterrows():
        sig = "***" if r['p_value'] < 0.01 else ("**" if r['p_value'] < 0.05 else ("*" if r['p_value'] < 0.1 else ""))
        print(f"  {r['bank']} Fold {r['fold']}: DM stat={r['dm_stat']:.4f}, p={r['p_value']:.4f} {sig} "
              f"(significant at 5%: {r['significant_05']})")

    print("\n" + "=" * 70)
    print("SENSITIVITY: SPLIT RATIO ANALYSIS")
    print("=" * 70)
    print(df_split[["bank", "split_ratio", "naive_mae", "garch_mae", "improvement_pct", "persistence"]].to_string(index=False))

    print(f"\nSaved to {OUTPUT_DIR}:")
    print("  - sensitivity_split_ratios.csv")
    print("  - garch_params_4fold.csv")
    print("  - diebold_mariano_results.csv")


if __name__ == "__main__":
    main()
