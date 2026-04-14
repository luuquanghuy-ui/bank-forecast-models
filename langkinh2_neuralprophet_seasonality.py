"""
Lăng kính 2: NeuralProphet — "Khi nào cổ phiếu ngân hàng VN có pattern?"

Part A: NP Decomposition (trend + seasonality) trên close price
Part B: Statistical calendar tests (Kruskal-Wallis, Mann-Whitney) trên return
Part C: So sánh NP seasonality vs Statistical test

Chạy: python langkinh2_neuralprophet_seasonality.py
Output: langkinh2_neuralprophet_seasonality/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from neuralprophet import NeuralProphet, set_log_level
    set_log_level("ERROR")
    print("NeuralProphet loaded")
except ImportError:
    print("ERROR: pip install neuralprophet")
    raise SystemExit(1)


# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "langkinh2_neuralprophet_seasonality"
OUTPUT_DIR.mkdir(exist_ok=True)

BANKS = ['BID', 'CTG', 'VCB']
BANK_FILES = {t: SCRIPT_DIR / f"banks_{t}_dataset.csv" for t in BANKS}

COLORS = {
    'BID': '#2E86AB',
    'CTG': '#A23B72',
    'VCB': '#F18F01',
}

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data(ticker):
    path = BANK_FILES[ticker]
    df = pd.read_csv(path, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Mon, 4=Fri
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    return df


# ============================================================
# 2. PART A — NeuralProphet Decomposition
# ============================================================
def run_np_decomposition(df, ticker):
    """Decompose close price into trend + weekly + yearly seasonality."""
    print(f"\n  Part A: NP Decomposition...")

    # Prepare data for NP
    np_df = df[['date', 'close']].copy()
    np_df.columns = ['ds', 'y']

    # Train NP with seasonality
    model = NeuralProphet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        n_lags=0,
        epochs=30,
        learning_rate=0.1,
        batch_size=64,
    )

    model.fit(np_df, freq='D')

    # Predict to get components
    forecast = model.predict(np_df)

    # Plot components
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Trend
    axes[0].plot(forecast['ds'], forecast['trend'], color=COLORS[ticker], linewidth=1.5)
    axes[0].set_ylabel('Trend')
    axes[0].set_title(f'{ticker} — Trend Component (long-term direction)')

    # Weekly seasonality
    if 'season_weekly' in forecast.columns:
        weekly = forecast.groupby(forecast['ds'].dt.dayofweek)['season_weekly'].mean()
        axes[1].bar(range(5), weekly.values[:5], color=COLORS[ticker], alpha=0.8)
        axes[1].set_xticks(range(5))
        axes[1].set_xticklabels(DAY_NAMES)
        axes[1].set_ylabel('Seasonal Effect')
        axes[1].set_title(f'{ticker} — Weekly Seasonality (NP decomposition)')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    else:
        axes[1].text(0.5, 0.5, 'No weekly seasonality component found',
                    transform=axes[1].transAxes, ha='center')

    # Yearly seasonality
    if 'season_yearly' in forecast.columns:
        monthly = forecast.copy()
        monthly['month'] = monthly['ds'].dt.month
        monthly_avg = monthly.groupby('month')['season_yearly'].mean()
        axes[2].bar(range(1, 13), monthly_avg.values, color=COLORS[ticker], alpha=0.8)
        axes[2].set_xticks(range(1, 13))
        axes[2].set_xticklabels(MONTH_NAMES)
        axes[2].set_ylabel('Seasonal Effect')
        axes[2].set_title(f'{ticker} — Yearly Seasonality (NP decomposition)')
        axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    else:
        axes[2].text(0.5, 0.5, 'No yearly seasonality component found',
                    transform=axes[2].transAxes, ha='center')

    plt.suptitle(f'{ticker} — NeuralProphet Decomposition (Close Price)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{ticker}_np_decomposition.png')
    plt.close()
    print(f"  Saved: {ticker}_np_decomposition.png")

    return forecast


# ============================================================
# 3. PART B — Statistical Calendar Tests
# ============================================================
def run_calendar_tests(df, ticker):
    """Statistical tests for calendar effects on daily return."""
    print(f"\n  Part B: Statistical Calendar Tests...")

    returns = df['log_return'].values
    results = []

    # --- Day-of-Week Effect ---
    day_groups = [df[df['day_of_week'] == d]['log_return'].values for d in range(5)]
    # Filter out empty groups (some days might not have trading)
    day_groups_clean = [g for g in day_groups if len(g) > 10]

    if len(day_groups_clean) >= 2:
        kw_stat, kw_p = stats.kruskal(*day_groups_clean)
    else:
        kw_stat, kw_p = np.nan, np.nan

    results.append({
        'test': 'Day-of-Week Effect',
        'method': 'Kruskal-Wallis',
        'statistic': kw_stat,
        'p_value': kw_p,
        'significant': kw_p < 0.05 if not np.isnan(kw_p) else False,
        'detail': f'H0: all days have same return distribution',
    })

    # Mean return per day
    day_means = {}
    day_stds = {}
    for d in range(5):
        day_ret = df[df['day_of_week'] == d]['log_return']
        day_means[DAY_NAMES[d]] = day_ret.mean()
        day_stds[DAY_NAMES[d]] = day_ret.std()

    # --- Monday Effect ---
    monday = df[df['day_of_week'] == 0]['log_return'].values
    other_days = df[df['day_of_week'] != 0]['log_return'].values
    if len(monday) > 10 and len(other_days) > 10:
        mw_stat, mw_p = stats.mannwhitneyu(monday, other_days, alternative='two-sided')
    else:
        mw_stat, mw_p = np.nan, np.nan

    results.append({
        'test': 'Monday Effect',
        'method': 'Mann-Whitney U',
        'statistic': mw_stat,
        'p_value': mw_p,
        'significant': mw_p < 0.05 if not np.isnan(mw_p) else False,
        'detail': f'Monday mean={df[df["day_of_week"]==0]["log_return"].mean():.6f}',
    })

    # --- Friday Effect ---
    friday = df[df['day_of_week'] == 4]['log_return'].values
    other_days_f = df[df['day_of_week'] != 4]['log_return'].values
    if len(friday) > 10 and len(other_days_f) > 10:
        fw_stat, fw_p = stats.mannwhitneyu(friday, other_days_f, alternative='two-sided')
    else:
        fw_stat, fw_p = np.nan, np.nan

    results.append({
        'test': 'Friday Effect',
        'method': 'Mann-Whitney U',
        'statistic': fw_stat,
        'p_value': fw_p,
        'significant': fw_p < 0.05 if not np.isnan(fw_p) else False,
        'detail': f'Friday mean={df[df["day_of_week"]==4]["log_return"].mean():.6f}',
    })

    # --- Month-of-Year Effect ---
    month_groups = [df[df['month'] == m]['log_return'].values for m in range(1, 13)]
    month_groups_clean = [g for g in month_groups if len(g) > 5]

    if len(month_groups_clean) >= 2:
        mk_stat, mk_p = stats.kruskal(*month_groups_clean)
    else:
        mk_stat, mk_p = np.nan, np.nan

    results.append({
        'test': 'Month-of-Year Effect',
        'method': 'Kruskal-Wallis',
        'statistic': mk_stat,
        'p_value': mk_p,
        'significant': mk_p < 0.05 if not np.isnan(mk_p) else False,
        'detail': 'H0: all months have same return distribution',
    })

    # Mean return per month
    month_means = {}
    for m in range(1, 13):
        month_ret = df[df['month'] == m]['log_return']
        month_means[MONTH_NAMES[m-1]] = month_ret.mean() if len(month_ret) > 0 else np.nan

    # --- January Effect ---
    jan = df[df['month'] == 1]['log_return'].values
    other_months = df[df['month'] != 1]['log_return'].values
    if len(jan) > 10 and len(other_months) > 10:
        jan_stat, jan_p = stats.mannwhitneyu(jan, other_months, alternative='two-sided')
    else:
        jan_stat, jan_p = np.nan, np.nan

    results.append({
        'test': 'January Effect',
        'method': 'Mann-Whitney U',
        'statistic': jan_stat,
        'p_value': jan_p,
        'significant': jan_p < 0.05 if not np.isnan(jan_p) else False,
        'detail': f'January mean={df[df["month"]==1]["log_return"].mean():.6f}',
    })

    # --- Quarter-End Effect ---
    # Last 5 trading days of each quarter
    df_copy = df.copy()
    df_copy['is_quarter_end'] = False
    for year in df_copy['year'].unique():
        for q in [1, 2, 3, 4]:
            end_month = q * 3
            mask = (df_copy['year'] == year) & (df_copy['month'] == end_month)
            quarter_data = df_copy[mask]
            if len(quarter_data) >= 5:
                last5_idx = quarter_data.index[-5:]
                df_copy.loc[last5_idx, 'is_quarter_end'] = True

    qe_ret = df_copy[df_copy['is_quarter_end']]['log_return'].values
    non_qe_ret = df_copy[~df_copy['is_quarter_end']]['log_return'].values
    if len(qe_ret) > 10 and len(non_qe_ret) > 10:
        qe_stat, qe_p = stats.mannwhitneyu(qe_ret, non_qe_ret, alternative='two-sided')
    else:
        qe_stat, qe_p = np.nan, np.nan

    results.append({
        'test': 'Quarter-End Effect',
        'method': 'Mann-Whitney U',
        'statistic': qe_stat,
        'p_value': qe_p,
        'significant': qe_p < 0.05 if not np.isnan(qe_p) else False,
        'detail': f'Quarter-end mean={np.mean(qe_ret):.6f}, Other mean={np.mean(non_qe_ret):.6f}',
    })

    # Print results
    print(f"\n  Calendar Test Results for {ticker}:")
    print(f"  {'Test':<25} {'p-value':<12} {'Significant?':<15}")
    print(f"  {'-'*52}")
    for r in results:
        sig_str = 'YES ***' if r['significant'] else 'no'
        p_str = f"{r['p_value']:.4f}" if not np.isnan(r['p_value']) else 'N/A'
        print(f"  {r['test']:<25} {p_str:<12} {sig_str:<15}")

    return results, day_means, day_stds, month_means


def plot_calendar_charts(df, ticker, day_means, month_means):
    """Boxplots for day-of-week and month-of-year returns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Day-of-week boxplot
    day_data = [df[df['day_of_week'] == d]['log_return'].values for d in range(5)]
    bp1 = axes[0].boxplot(day_data, labels=DAY_NAMES, patch_artist=True, showfliers=False)
    for patch in bp1['boxes']:
        patch.set_facecolor(COLORS[ticker])
        patch.set_alpha(0.6)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Daily Log Return')
    axes[0].set_title(f'{ticker} — Return by Day of Week')

    # Add mean markers
    means = [np.mean(d) for d in day_data]
    axes[0].plot(range(1, 6), means, 'D', color='red', markersize=6, label='Mean')
    axes[0].legend()

    # Month boxplot
    month_data = [df[df['month'] == m]['log_return'].values for m in range(1, 13)]
    bp2 = axes[1].boxplot(month_data, labels=MONTH_NAMES, patch_artist=True, showfliers=False)
    for patch in bp2['boxes']:
        patch.set_facecolor(COLORS[ticker])
        patch.set_alpha(0.6)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Daily Log Return')
    axes[1].set_title(f'{ticker} — Return by Month')
    axes[1].tick_params(axis='x', rotation=45)

    # Add mean markers
    means_m = [np.mean(d) if len(d) > 0 else 0 for d in month_data]
    axes[1].plot(range(1, 13), means_m, 'D', color='red', markersize=5, label='Mean')
    axes[1].legend()

    plt.suptitle(f'{ticker} — Calendar Effect Analysis (Part B)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{ticker}_calendar_boxplots.png')
    plt.close()
    print(f"  Saved: {ticker}_calendar_boxplots.png")


# ============================================================
# 4. CROSS-BANK COMPARISON
# ============================================================
def plot_cross_bank_calendar(all_day_means, all_month_means):
    """Compare calendar patterns across 3 banks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Day-of-week comparison
    for ticker in BANKS:
        means = [all_day_means[ticker][d] for d in DAY_NAMES]
        axes[0].plot(DAY_NAMES, means, 'o-', label=ticker, color=COLORS[ticker], linewidth=2, markersize=8)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Mean Daily Return')
    axes[0].set_title('Day-of-Week: Mean Return Comparison')
    axes[0].legend()

    # Month comparison
    for ticker in BANKS:
        means = [all_month_means[ticker].get(m, 0) for m in MONTH_NAMES]
        axes[1].plot(MONTH_NAMES, means, 'o-', label=ticker, color=COLORS[ticker], linewidth=2, markersize=6)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Mean Daily Return')
    axes[1].set_title('Month-of-Year: Mean Return Comparison')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle('Cross-Bank Calendar Pattern Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'crossbank_calendar_comparison.png')
    plt.close()
    print(f"  Saved: crossbank_calendar_comparison.png")


def plot_significance_summary(all_results):
    """Heatmap of p-values across banks and tests."""
    tests = [r['test'] for r in all_results[BANKS[0]]]
    n_tests = len(tests)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Build p-value matrix
    p_matrix = np.zeros((n_tests, len(BANKS)))
    for j, ticker in enumerate(BANKS):
        for i, r in enumerate(all_results[ticker]):
            p_matrix[i, j] = r['p_value'] if not np.isnan(r['p_value']) else 1.0

    # Heatmap
    im = ax.imshow(p_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.2)

    ax.set_xticks(range(len(BANKS)))
    ax.set_xticklabels(BANKS, fontsize=12, fontweight='bold')
    ax.set_yticks(range(n_tests))
    ax.set_yticklabels(tests, fontsize=10)

    # Annotate
    for i in range(n_tests):
        for j in range(len(BANKS)):
            p = p_matrix[i, j]
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
            text = f'{p:.3f}{sig}'
            color = 'white' if p < 0.05 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=10,
                    fontweight='bold' if p < 0.05 else 'normal', color=color)

    plt.colorbar(im, label='p-value (green=significant, red=not)', shrink=0.8)
    ax.set_title('Calendar Effect Significance (p-values)\n*** p<0.01, ** p<0.05, * p<0.10',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'significance_heatmap.png')
    plt.close()
    print(f"  Saved: significance_heatmap.png")


# ============================================================
# 5. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("LANG KINH 2: NeuralProphet — Calendar Effects")
    print("Cau hoi: Khi nao co phieu ngan hang VN co pattern?")
    print("=" * 60)

    all_results = {}
    all_day_means = {}
    all_month_means = {}

    for ticker in BANKS:
        print(f"\n{'#'*50}")
        print(f"# {ticker}")
        print(f"{'#'*50}")

        df = load_data(ticker)
        print(f"  Data: {len(df)} rows, {df['date'].min().date()} -> {df['date'].max().date()}")

        # Part A: NP Decomposition
        forecast = run_np_decomposition(df, ticker)

        # Part B: Statistical Calendar Tests
        results, day_means, day_stds, month_means = run_calendar_tests(df, ticker)
        all_results[ticker] = results
        all_day_means[ticker] = day_means
        all_month_means[ticker] = month_means

        # Calendar boxplots
        plot_calendar_charts(df, ticker, day_means, month_means)

        # Save results CSV
        pd.DataFrame(results).to_csv(OUTPUT_DIR / f'{ticker}_calendar_tests.csv', index=False)

    # ===== CROSS-BANK =====
    print(f"\n{'='*60}")
    print("CROSS-BANK ANALYSIS")
    print(f"{'='*60}")

    plot_cross_bank_calendar(all_day_means, all_month_means)
    plot_significance_summary(all_results)

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*60}")
    print("KET QUA CHINH")
    print(f"{'='*60}")

    print("\nCalendar Effect Summary:")
    print(f"  {'Test':<25}", end="")
    for t in BANKS:
        print(f" {t:<12}", end="")
    print()
    print(f"  {'-'*61}")

    test_names = [r['test'] for r in all_results[BANKS[0]]]
    for i, test in enumerate(test_names):
        print(f"  {test:<25}", end="")
        for ticker in BANKS:
            r = all_results[ticker][i]
            p = r['p_value']
            if np.isnan(p):
                print(f" {'N/A':<12}", end="")
            elif p < 0.01:
                print(f" {p:.4f} ***  ", end="")
            elif p < 0.05:
                print(f" {p:.4f} **   ", end="")
            elif p < 0.10:
                print(f" {p:.4f} *    ", end="")
            else:
                print(f" {p:.4f}      ", end="")
        print()

    # Day-of-week means
    print("\nMean Return by Day:")
    print(f"  {'Day':<8}", end="")
    for t in BANKS:
        print(f" {t:<12}", end="")
    print()
    for d in DAY_NAMES:
        print(f"  {d:<8}", end="")
        for t in BANKS:
            print(f" {all_day_means[t][d]:>+.6f}  ", end="")
        print()

    # Best/worst months
    print("\nBest & Worst Months (mean daily return):")
    for ticker in BANKS:
        sorted_months = sorted(all_month_means[ticker].items(), key=lambda x: x[1] if not np.isnan(x[1]) else 0)
        worst = sorted_months[0]
        best = sorted_months[-1]
        print(f"  {ticker}: Best={best[0]} ({best[1]:+.6f}), Worst={worst[0]} ({worst[1]:+.6f})")

    print(f"\n\nAll output saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
