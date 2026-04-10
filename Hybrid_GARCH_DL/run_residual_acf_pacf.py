"""
Residual ACF/PACF Analysis for GARCH(1,1)

Check if standardized residuals are i.i.d.:
- ACF should show no significant autocorrelations
- PACF should show no significant partial autocorrelations
- Ljung-Box test should NOT reject i.i.d. hypothesis

If residuals show autocorrelation, GARCH(1,1) is misspecified.
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "residual_acf_pacf_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BANK_FILES = {
    "BID": BASE_DIR / "banks_BID_dataset.csv",
    "CTG": BASE_DIR / "banks_CTG_dataset.csv",
    "VCB": BASE_DIR / "banks_VCB_dataset.csv",
}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"date": "ds", "close": "y"})
    df["log_return"] = np.log(df["y"] / df["y"].shift(1))
    return df.dropna().reset_index(drop=True)


def split_data(df):
    n = len(df)
    return df.iloc[:int(n*0.70)].copy(), df.iloc[int(n*0.70):int(n*0.85)].copy(), df.iloc[int(n*0.85):].copy()


def garch_fit_and_residuals(ret_scaled):
    """Fit GARCH(1,1) and return standardized residuals."""
    model = arch_model(ret_scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = model.fit(disp="off", show_warning=False, options={"maxiter": 1000})

    # Standardized residuals - may be numpy array or pandas Series
    std_resid_raw = res.std_resid
    if hasattr(std_resid_raw, 'dropna'):
        std_resid = std_resid_raw.dropna().values
    else:
        std_resid = np.array(std_resid_raw)
        std_resid = std_resid[~np.isnan(std_resid)]

    return {
        "params": res.params,
        "std_resid": std_resid,
        "model": res
    }


def main():
    print("=" * 70)
    print("RESIDUAL ACF/PACF ANALYSIS - GARCH(1,1) MODEL DIAGNOSTICS")
    print("=" * 70)

    all_results = []
    max_lags = 20

    for bank, path in BANK_FILES.items():
        print(f"\n{'='*50}\n{bank}\n{'='*50}")

        df = load_data(path)
        train, val, test = split_data(df)

        train_ret = train["log_return"].values
        train_scaled = train_ret * 100.0

        # Fit GARCH on training data
        result = garch_fit_and_residuals(train_scaled)
        std_resid = result["std_resid"]
        params = result["params"]

        print(f"  Training size: {len(train_ret)} days")
        print(f"  Std residuals: {len(std_resid)} values")
        print(f"\n  GARCH(1,1) Parameters:")
        for k, v in params.items():
            print(f"    {k}: {v:.6f}")

        # Ljung-Box test for autocorrelation
        # H0: no autocorrelation in residuals
        lb_test = acorr_ljungbox(std_resid, lags=[5, 10, 15, 20], return_df=True)

        print(f"\n  Ljung-Box Test (H0: no autocorrelation):")
        for lag in [5, 10, 15, 20]:
            lb_stat = lb_test.loc[lag, "lb_stat"]
            lb_pvalue = lb_test.loc[lag, "lb_pvalue"]
            significance = "***" if lb_pvalue < 0.001 else "**" if lb_pvalue < 0.01 else "*" if lb_pvalue < 0.05 else ""
            print(f"    Lag {lag:2d}: Q={lb_stat:8.4f}, p-value={lb_pvalue:.4f} {significance}")

        # Check if residuals are Gaussian (Jarque-Bera like)
        resid_skew = pd.Series(std_resid).skew()
        resid_kurt = pd.Series(std_resid).kurtosis()
        print(f"\n  Residual Diagnostics:")
        print(f"    Skewness: {resid_skew:.4f} (should be ~0)")
        print(f"    Excess Kurtosis: {resid_kurt:.4f} (should be ~0 for normal)")

        # Save Ljung-Box results
        lb_summary = lb_test.copy()
        lb_summary["bank"] = bank
        all_results.append({
            "bank": bank,
            "lb_lag5_stat": lb_test.loc[5, "lb_stat"],
            "lb_lag5_pvalue": lb_test.loc[5, "lb_pvalue"],
            "lb_lag10_stat": lb_test.loc[10, "lb_stat"],
            "lb_lag10_pvalue": lb_test.loc[10, "lb_pvalue"],
            "lb_lag20_stat": lb_test.loc[20, "lb_stat"],
            "lb_lag20_pvalue": lb_test.loc[20, "lb_pvalue"],
            "skewness": resid_skew,
            "kurtosis": resid_kurt
        })

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: ACF of standardized residuals
        ax1 = axes[0, 0]
        plot_acf(std_resid, ax=ax1, lags=max_lags, alpha=0.05)
        ax1.set_title(f"{bank} - ACF of Standardized Residuals")
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("Autocorrelation")

        # Plot 2: PACF of standardized residuals
        ax2 = axes[0, 1]
        plot_pacf(std_resid, ax=ax2, lags=max_lags, alpha=0.05, method='ywm')
        ax2.set_title(f"{bank} - PACF of Standardized Residuals")
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("Partial Autocorrelation")

        # Plot 3: Histogram of residuals
        ax3 = axes[1, 0]
        ax3.hist(std_resid, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black")
        x = np.linspace(-4, 4, 100)
        from scipy import stats
        ax3.plot(x, stats.norm.pdf(x), 'r-', lw=2, label="Normal")
        ax3.set_title(f"{bank} - Distribution of Standardized Residuals")
        ax3.set_xlabel("Standardized Residual")
        ax3.set_ylabel("Density")
        ax3.legend()

        # Plot 4: Q-Q plot
        ax4 = axes[1, 1]
        stats.probplot(std_resid, dist="norm", plot=ax4)
        ax4.set_title(f"{bank} - Q-Q Plot (Normal)")
        ax4.get_lines()[0].set_markerfacecolor('steelblue')
        ax4.get_lines()[0].set_markeredgecolor('steelblue')
        ax4.get_lines()[1].set_color('red')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{bank}_residual_acf_pacf.png", dpi=150)
        plt.close()
        print(f"\n  Chart saved: {bank}_residual_acf_pacf.png")

    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_DIR / "residual_diagnostics.csv", index=False)

    print("\n" + "=" * 70)
    print("RESIDUAL DIAGNOSTICS SUMMARY")
    print("=" * 70)
    print("\nInterpretation:")
    print("- If Ljung-Box p-value > 0.05: Cannot reject H0 (residuals are i.i.d.) - GOOD")
    print("- If Ljung-Box p-value < 0.05: Reject H0 (residuals have autocorrelation) - BAD")
    print("- Skewness ~0 and Kurtosis ~0: Residuals are approximately Gaussian - GOOD\n")

    for _, r in df_results.iterrows():
        print(f"{r['bank']}:")
        print(f"  Lag 10: Q={r['lb_lag10_stat']:.4f}, p={r['lb_lag10_pvalue']:.4f} {'(i.i.d.)' if r['lb_lag10_pvalue'] > 0.05 else '(autocorrelated!)'}")
        print(f"  Lag 20: Q={r['lb_lag20_stat']:.4f}, p={r['lb_lag20_pvalue']:.4f} {'(i.i.d.)' if r['lb_lag20_pvalue'] > 0.05 else '(autocorrelated!)'}")

    print(f"\n\nSaved to: {OUTPUT_DIR}/")
    print(f"  - residual_diagnostics.csv")
    print(f"  - BID_residual_acf_pacf.png")
    print(f"  - CTG_residual_acf_pacf.png")
    print(f"  - VCB_residual_acf_pacf.png")


if __name__ == "__main__":
    main()