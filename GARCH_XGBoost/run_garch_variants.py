"""
EGARCH and GJR-GARCH comparison with GARCH(1,1)

Asymmetric volatility: negative shocks increase volatility more than positive shocks.
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
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

OUTPUT_DIR = SCRIPT_DIR / "garch_variants_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"date": "ds", "close": "y"})
    df["log_return"] = np.log(df["y"] / df["y"].shift(1))
    return df.dropna().reset_index(drop=True)


def split_data(df):
    n = len(df)
    return df.iloc[:int(n*0.70)].copy(), df.iloc[int(n*0.70):int(n*0.85)].copy(), df.iloc[int(n*0.85):].copy()


def garch_walkforward(train_ret, test_ret, variant="GARCH"):
    """Walk-forward for GARCH variants."""
    ret_scaled = train_ret * 100.0

    # Replace any NaN/inf with small values
    ret_scaled = np.nan_to_num(ret_scaled, nan=0.0, posinf=1e-6, neginf=-1e-6)

    if variant == "GARCH":
        model = arch_model(ret_scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    elif variant == "GJR-GARCH":
        model = arch_model(ret_scaled, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal", rescale=False)
    elif variant == "EGARCH":
        model = arch_model(ret_scaled, mean="Constant", vol="EGARCH", p=1, q=1, dist="normal", rescale=False)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Try fitting with different options for convergence
    try:
        res = model.fit(disp="off", show_warning=False, options={"maxiter": 1000})
    except Exception:
        # Try with different optimizer
        try:
            res = model.fit(disp="off", show_warning=False, options={"maxiter": 2000, "xtol": 1e-8})
        except Exception:
            return None, None

    # Check convergence - use convergence_flag attribute
    if hasattr(res, 'convergence_flag') and res.convergence_flag != 0:
        return None, None

    mu = float(res.params.get("mu", res.params.get("Const", 0.0))) / 100.0

    # Get params - handle different naming conventions
    omega = float(res.params["omega"])

    if variant == "GARCH":
        alpha = float(res.params["alpha[1]"])
        beta = float(res.params["beta[1]"])
        gamma = np.nan  # Not applicable for GARCH
    elif variant == "GJR-GARCH":
        alpha = float(res.params["alpha[1]"])
        gamma_key = "gamma[1]" if "gamma[1]" in res.params else "leverage[1]"
        gamma = float(res.params.get(gamma_key, 0.0))
        beta = float(res.params["beta[1]"])
    elif variant == "EGARCH":
        # EGARCH in arch library: log(σ²) = ω + α*(|z|-√(2/π)) + β*log(σ²_{t-1})
        # No separate gamma - asymmetry is embedded in arch's parameterization
        alpha = float(res.params["alpha[1]"])
        beta = float(res.params["beta[1]"])
        gamma = np.nan  # EGARCH in arch has no separate gamma parameter

    # Walk-forward prediction
    sigma2 = np.empty(len(test_ret))
    sigma2_last = max(float(np.var(ret_scaled)), 1e-6)
    eps_last = (test_ret[0] - mu) * 100.0

    for i in range(len(test_ret)):
        if variant == "GARCH":
            sigma2_raw = omega + alpha * (eps_last ** 2) + beta * sigma2_last
        elif variant == "GJR-GARCH":
            # Leverage effect: negative shocks have gamma added
            shock = gamma * eps_last if eps_last < 0 else 0
            sigma2_raw = omega + (alpha + shock) * (eps_last ** 2) + beta * sigma2_last
        elif variant == "EGARCH":
            # EGARCH log-volatility form per arch library:
            # z = eps/sigma, so |eps|/sqrt(sigma2_last) = |z|
            z = eps_last / np.sqrt(sigma2_last) if sigma2_last > 0 else 0
            log_sigma2 = omega + alpha * (np.abs(z) - np.sqrt(2/np.pi)) + beta * np.log(sigma2_last)
            sigma2_raw = np.exp(log_sigma2)

        # Floor sigma2 to prevent negative variance
        sigma2[i] = max(sigma2_raw, 1e-8)
        eps_last = (test_ret[i] - mu) * 100.0
        sigma2_last = sigma2[i]

    # Return sigma, result, and gamma (for GJR-GARCH leverage analysis)
    gamma_val = gamma if variant == "GJR-GARCH" else np.nan
    return np.sqrt(sigma2) / 100.0, res, gamma_val


def main():
    print("=" * 70)
    print("GARCH VARIANTS COMPARISON: GARCH(1,1) vs GJR-GARCH(1,1) vs EGARCH(1,1)")
    print("=" * 70)

    variants = ["GARCH", "GJR-GARCH", "EGARCH"]
    all_results = []

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        df = load_data(path)
        train, val, test = split_data(df)

        train_ret = train["log_return"].values
        test_ret = test["log_return"].values
        test_abs = np.abs(test_ret)

        # Naive
        naive_pred = np.zeros(len(test_ret))
        naive_mae = mean_absolute_error(test_abs, naive_pred)

        print(f"  Naive MAE: {naive_mae:.6f}")

        variant_preds = {}
        variant_params = {}

        for variant in variants:
            try:
                garch_sigma, res, gamma_val = garch_walkforward(train_ret, test_ret, variant)
                if garch_sigma is None or res is None:
                    print(f"  {variant}: FAILED - did not converge")
                    continue
                garch_pred = np.sqrt(2 / np.pi) * garch_sigma
                mae = mean_absolute_error(test_abs, garch_pred)
                variant_preds[variant] = garch_pred
                # Store params with gamma_used
                variant_params[variant] = {**res.params.to_dict(), "gamma_used": gamma_val}

                print(f"  {variant}: MAE={mae:.6f}")

                # Store params for CSV
                params_dict = {"bank": bank, "variant": variant, "mae": mae}
                for k, v in res.params.items():
                    params_dict[k] = float(v)
                for k, v in res.std_err.items():
                    params_dict[f"{k}_se"] = float(v)
                params_dict["gamma_used"] = gamma_val

                all_results.append(params_dict)

            except Exception as e:
                print(f"  {variant}: FAILED - {e}")

        # Compare: which variant is best?
        best_variant = min(variant_preds.keys(), key=lambda v: mean_absolute_error(test_abs, variant_preds[v]))
        best_mae = mean_absolute_error(test_abs, variant_preds[best_variant])
        print(f"\n  BEST: {best_variant} (MAE={best_mae:.6f})")

        # Check leverage effect in GJR-GARCH
        if "GJR-GARCH" in variant_params:
            gamma = float(variant_params["GJR-GARCH"]["gamma_used"])
            gamma_key = "gamma[1]" if "gamma[1]" in res.params else "leverage[1]"
            gamma_se = float(res.std_err.get(gamma_key, 1.0))
            gamma_z = gamma / gamma_se if gamma_se > 0 else 0
            gamma_p = 2 * (1 - stats.norm.cdf(abs(gamma_z)))
            print(f"\n  GJR-GARCH Leverage Effect:")
            print(f"    gamma = {gamma:.4f} (z={gamma_z:.2f}, p={gamma_p:.4f})")
            if gamma > 0:
                print(f"    -> Negative shocks increase volatility MORE than positive")
            else:
                print(f"    -> Positive shocks increase volatility MORE than negative")

        # Note about EGARCH asymmetry
        if "EGARCH" in variant_params:
            egarch_alpha = float(variant_params["EGARCH"]["alpha[1]"])
            print(f"\n  EGARCH Note:")
            print(f"    alpha = {egarch_alpha:.4f}")
            print(f"    (EGARCH asymmetry in arch library is embedded in alpha, no separate gamma)")

    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_DIR / "garch_variants_comparison.csv", index=False)

    print(f"\n\nSaved: {OUTPUT_DIR / 'garch_variants_comparison.csv'}")


if __name__ == "__main__":
    main()
