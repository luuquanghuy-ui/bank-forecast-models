"""
Lang kinh 3: TFT + ACF — "Bao xa tri nho thi truong?"

Part A: ACF/PACF (thong ke truyen thong) tren return va |return|
Part B: TFT Attention — train TFT, extract attention weights
Part C: Cross-bank comparison — ACF vs TFT Attention

Chay: .venv-neural/Scripts/python langkinh3_tft_memory.py
Output: langkinh3_tft_memory/
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

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# TFT imports
try:
    import torch
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from lightning.pytorch import Trainer
    HAS_TFT = True
    print(f"TFT available: torch={torch.__version__}")
except ImportError:
    HAS_TFT = False
    print("WARNING: pytorch_forecasting not available, skipping TFT attention")


# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "langkinh3_tft_memory"
OUTPUT_DIR.mkdir(exist_ok=True)

BANKS = ['BID', 'CTG', 'VCB']
BANK_FILES = {t: SCRIPT_DIR / f"banks_{t}_dataset.csv" for t in BANKS}

COLORS = {
    'BID': '#2E86AB',
    'CTG': '#A23B72',
    'VCB': '#F18F01',
}

MAX_LAGS = 60  # 3 months of trading days

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
    return df


# ============================================================
# 2. PART A — ACF / PACF
# ============================================================
def compute_acf_pacf(series, nlags=MAX_LAGS):
    """Compute ACF and PACF with confidence intervals."""
    acf_vals, acf_ci = acf(series, nlags=nlags, alpha=0.05)
    pacf_vals, pacf_ci = pacf(series, nlags=nlags, alpha=0.05)
    return acf_vals, acf_ci, pacf_vals, pacf_ci


def count_significant_lags(acf_vals, n_obs, start=1):
    """Count how many lags have ACF outside 95% confidence interval."""
    ci_bound = 1.96 / np.sqrt(n_obs)
    n_sig = 0
    first_insignificant = None
    for i in range(start, len(acf_vals)):
        if abs(acf_vals[i]) > ci_bound:
            n_sig += 1
        elif first_insignificant is None:
            first_insignificant = i
    return n_sig, first_insignificant


def plot_acf_pacf_single(df, ticker):
    """Plot ACF/PACF for return and |return| for one bank."""
    ret = df['log_return'].dropna().values
    abs_ret = np.abs(ret)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    series_list = [
        (ret, 'Log Return', 'Return hom nay lien quan bao nhieu ngay truoc?'),
        (abs_ret, '|Log Return| (proxy volatility)', 'Bien dong hom nay lien quan bao nhieu ngay truoc?'),
    ]

    for row, (series, name, question) in enumerate(series_list):
        acf_vals, acf_ci, pacf_vals, pacf_ci = compute_acf_pacf(series)

        # ACF plot
        ax = axes[row, 0]
        ax.bar(range(len(acf_vals)), acf_vals, width=0.4, color=COLORS[ticker], alpha=0.7)
        n = len(series)
        ci_bound = 1.96 / np.sqrt(n)
        ax.axhline(y=ci_bound, color='red', linestyle='--', alpha=0.5, label=f'95% CI (+-{ci_bound:.3f})')
        ax.axhline(y=-ci_bound, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_xlabel('Lag (ngay)')
        ax.set_ylabel('ACF')
        ax.set_title(f'{ticker} ACF — {name}')
        ax.set_xlim(-0.5, MAX_LAGS + 0.5)
        ax.legend(fontsize=9)

        # PACF plot
        ax = axes[row, 1]
        ax.bar(range(len(pacf_vals)), pacf_vals, width=0.4, color=COLORS[ticker], alpha=0.7)
        ax.axhline(y=ci_bound, color='red', linestyle='--', alpha=0.5, label=f'95% CI')
        ax.axhline(y=-ci_bound, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_xlabel('Lag (ngay)')
        ax.set_ylabel('PACF')
        ax.set_title(f'{ticker} PACF — {name}')
        ax.set_xlim(-0.5, MAX_LAGS + 0.5)
        ax.legend(fontsize=9)

    plt.suptitle(f'{ticker} — ACF/PACF Analysis\n'
                 f'Tren: return (gia tang/giam) | Duoi: |return| (muc do bien dong)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{ticker}_acf_pacf.png')
    plt.close()
    print(f"  Saved: {ticker}_acf_pacf.png")


def analyze_memory(df, ticker):
    """Analyze memory length for return and absolute return."""
    ret = df['log_return'].dropna().values
    abs_ret = np.abs(ret)

    results = {}

    for name, series in [('return', ret), ('abs_return', abs_ret)]:
        acf_vals, acf_ci, pacf_vals, pacf_ci = compute_acf_pacf(series)
        n_sig, first_insig = count_significant_lags(acf_vals, len(series))

        # Ljung-Box test at different lags
        lb_results = acorr_ljungbox(series, lags=[5, 10, 20, 40], return_df=True)

        results[name] = {
            'acf_vals': acf_vals,
            'n_significant_lags': n_sig,
            'first_insignificant_lag': first_insig,
            'acf_lag1': acf_vals[1] if len(acf_vals) > 1 else np.nan,
            'acf_lag5': acf_vals[5] if len(acf_vals) > 5 else np.nan,
            'acf_lag10': acf_vals[10] if len(acf_vals) > 10 else np.nan,
            'acf_lag20': acf_vals[20] if len(acf_vals) > 20 else np.nan,
            'ljungbox': lb_results,
        }

        print(f"\n  {ticker} — {name}:")
        print(f"    ACF(1)  = {acf_vals[1]:.4f}")
        print(f"    ACF(5)  = {acf_vals[5]:.4f}")
        print(f"    ACF(10) = {acf_vals[10]:.4f}")
        print(f"    ACF(20) = {acf_vals[20]:.4f}")
        print(f"    Significant lags (out of {MAX_LAGS}): {n_sig}")
        print(f"    First insignificant lag: {first_insig}")

        print(f"    Ljung-Box test:")
        for lag_val in [5, 10, 20, 40]:
            row = lb_results.loc[lag_val]
            p = row['lb_pvalue']
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else '')
            print(f"      Lag {lag_val:2d}: p={p:.6f} {sig}")

    return results


# ============================================================
# 3. Half-life
# ============================================================
def compute_halflife(acf_vals):
    """Compute half-life: at which lag ACF drops to half of ACF(1)."""
    if len(acf_vals) < 2 or acf_vals[1] <= 0:
        return None
    target = acf_vals[1] / 2
    for lag in range(2, len(acf_vals)):
        if acf_vals[lag] <= target:
            return lag
    return None


# ============================================================
# 4. TFT ATTENTION — Part B
# ============================================================
def train_tft_and_get_attention(df, ticker, lookback=24):
    """Train TFT on close price and extract attention weights."""
    if not HAS_TFT:
        print("  SKIP: TFT not available")
        return None

    print(f"  Training TFT (lookback={lookback})...")

    train_d = df.copy()
    train_d["time_idx"] = range(len(train_d))
    train_d["bank"] = ticker

    # Use 85% for training
    train_end = int(len(train_d) * 0.85)
    train_subset = train_d.iloc[:train_end].copy()

    training = TimeSeriesDataSet(
        train_subset,
        time_idx="time_idx",
        target="close",
        group_ids=["bank"],
        max_encoder_length=lookback,
        max_prediction_length=1,
        static_categoricals=["bank"],
        time_varying_unknown_reals=["close"],
        scalers={},
    )

    train_dataloader = training.to_dataloader(batch_size=64, num_workers=0, shuffle=False)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        learning_rate=0.001,
        optimizer="adam",
    )

    trainer = Trainer(
        max_epochs=15,
        accelerator="cpu",
        enable_progress_bar=False,
        logger=False,
    )

    trainer.fit(tft, train_dataloader)

    # Get attention on full data
    full_dataset = TimeSeriesDataSet.from_dataset(training, train_d, predict=False)
    full_dataloader = full_dataset.to_dataloader(batch_size=64, num_workers=0, shuffle=False)

    # Extract attention weights
    try:
        raw_output = tft.predict(full_dataloader, mode="raw", return_x=True)
        # raw_output is a tuple: (output, x)
        # output has .attention field
        output = raw_output[0]
        
        if hasattr(output, 'encoder_attention'):
            # attention shape: (n_samples, n_heads, encoder_length)
            attention_raw = output.encoder_attention.cpu().detach().numpy()
            # Average over all dims except last (encoder_length)
            # Shape could be (samples, 1, heads, lookback) or similar
            while attention_raw.ndim > 1:
                attention_raw = attention_raw.mean(axis=0)
            attention = attention_raw
            print(f"  TFT attention extracted: shape={attention.shape}")
            return attention
        else:
            # Try accessing via dict-like
            for attr in ['attention', 'attention_weights']:
                if hasattr(output, attr):
                    att = getattr(output, attr)
                    if att is not None:
                        attention = att.cpu().detach().numpy()
                        if attention.ndim > 1:
                            attention = attention.mean(axis=tuple(range(attention.ndim - 1)))
                        print(f"  TFT attention ({attr}): shape={attention.shape}")
                        return attention
            
            # Last resort: list all fields
            print(f"  Output type: {type(output)}")
            if hasattr(output, '_fields'):
                print(f"  Fields: {output._fields}")
            elif hasattr(output, '__dict__'):
                print(f"  Attrs: {list(output.__dict__.keys())}")
            print("  WARNING: Could not find attention in output")
            return None
    except Exception as e:
        print(f"  WARNING: Attention extraction failed: {e}")
        return None


def plot_tft_attention(all_attention, lookback=24):
    """Plot TFT attention weights as bar chart per bank."""
    available = {t: att for t, att in all_attention.items() if att is not None}
    if not available:
        print("  No TFT attention to plot")
        return

    n_banks = len(available)
    fig, axes = plt.subplots(1, n_banks, figsize=(6 * n_banks, 5))
    if n_banks == 1:
        axes = [axes]

    for i, (ticker, attention) in enumerate(available.items()):
        # attention may be 1D (averaged) or 2D
        if attention.ndim > 1:
            att = attention.mean(axis=0)  # average over samples
        else:
            att = attention

        n_steps = min(len(att), lookback)
        lags = list(range(n_steps, 0, -1))  # lookback labels: 24, 23, ..., 1
        att_plot = att[:n_steps]

        axes[i].bar(range(n_steps), att_plot, color=COLORS[ticker], alpha=0.8)
        axes[i].set_xlabel('Lookback step (ngay truoc)')
        axes[i].set_ylabel('Attention weight')
        axes[i].set_title(f'{ticker} — TFT Attention')

        # Mark top 3
        top3_idx = np.argsort(att_plot)[-3:]
        for idx in top3_idx:
            axes[i].bar(idx, att_plot[idx], color='red', alpha=0.9)

        # X-axis labels for every 4th step
        tick_positions = list(range(0, n_steps, 4))
        tick_labels = [str(n_steps - p) for p in tick_positions]
        axes[i].set_xticks(tick_positions)
        axes[i].set_xticklabels(tick_labels)

    plt.suptitle('Part B: TFT Attention — Model nhìn vào ngày nào nhiều nhất?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'partB_tft_attention.png')
    plt.close()
    print(f"  Saved: partB_tft_attention.png")


def plot_acf_vs_attention(all_memory, all_attention, lookback=24):
    """Part C: Compare ACF with TFT attention side by side."""
    available = {t: att for t, att in all_attention.items() if att is not None}
    if not available:
        print("  No TFT attention for comparison")
        return

    fig, axes = plt.subplots(2, len(BANKS), figsize=(6 * len(BANKS), 8))

    for j, ticker in enumerate(BANKS):
        # Row 0: ACF |return|
        acf_vals = all_memory[ticker]['abs_return']['acf_vals']
        axes[0, j].bar(range(1, lookback + 1), acf_vals[1:lookback+1],
                       color=COLORS[ticker], alpha=0.7)
        ci = 1.96 / np.sqrt(2500)
        axes[0, j].axhline(y=ci, color='red', linestyle='--', alpha=0.4)
        axes[0, j].axhline(y=-ci, color='red', linestyle='--', alpha=0.4)
        axes[0, j].set_title(f'{ticker} — ACF |Return|')
        axes[0, j].set_xlabel('Lag')
        axes[0, j].set_ylabel('ACF')

        # Row 1: TFT attention
        if ticker in available:
            att = available[ticker]
            if att.ndim > 1:
                att = att.mean(axis=0)
            n_steps = min(len(att), lookback)
            axes[1, j].bar(range(1, n_steps + 1), att[:n_steps][::-1],
                          color=COLORS[ticker], alpha=0.7)
            axes[1, j].set_title(f'{ticker} — TFT Attention')
        else:
            axes[1, j].text(0.5, 0.5, 'N/A', transform=axes[1, j].transAxes, ha='center')
            axes[1, j].set_title(f'{ticker} — TFT Attention (N/A)')
        axes[1, j].set_xlabel('Lag')
        axes[1, j].set_ylabel('Weight')

    plt.suptitle('Part C: ACF (thong ke) vs TFT Attention (deep learning)\n'
                 'Tren: ACF do tuong quan thuc | Duoi: TFT attention (model hoc duoc)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'partC_acf_vs_attention.png')
    plt.close()
    print(f"  Saved: partC_acf_vs_attention.png")


# ============================================================
# 5. CROSS-BANK COMPARISON
# ============================================================
def plot_crossbank_acf(all_memory):
    """Compare ACF across 3 banks for return and |return|."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, target in enumerate(['return', 'abs_return']):
        title = 'Log Return' if target == 'return' else '|Log Return| (Volatility proxy)'
        for ticker in BANKS:
            acf_vals = all_memory[ticker][target]['acf_vals']
            axes[i].plot(range(1, len(acf_vals)), acf_vals[1:],
                        '-', label=ticker, color=COLORS[ticker], linewidth=1.5, alpha=0.8)

        n = 2500
        ci = 1.96 / np.sqrt(n)
        axes[i].axhline(y=ci, color='red', linestyle='--', alpha=0.4, label='95% CI')
        axes[i].axhline(y=-ci, color='red', linestyle='--', alpha=0.4)
        axes[i].axhline(y=0, color='gray', linewidth=0.5)
        axes[i].set_xlabel('Lag (ngay)')
        axes[i].set_ylabel('ACF')
        axes[i].set_title(f'Cross-Bank ACF — {title}')
        axes[i].legend()
        axes[i].set_xlim(0, MAX_LAGS)

    plt.suptitle('Cross-Bank Memory Comparison\n'
                 'Trai: Return (random walk?) | Phai: |Return| (volatility clustering?)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'crossbank_acf_comparison.png')
    plt.close()
    print(f"  Saved: crossbank_acf_comparison.png")


def plot_memory_summary(all_memory):
    """Bar chart comparing memory metrics across banks."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, target in enumerate(['return', 'abs_return']):
        data = [all_memory[t][target]['n_significant_lags'] for t in BANKS]
        bars = axes[i].bar(BANKS, data, color=[COLORS[t] for t in BANKS], alpha=0.8)
        axes[i].set_ylabel('Significant lags (out of 60)')
        name = 'Return' if target == 'return' else '|Return|'
        axes[i].set_title(f'Memory Length — {name}')
        for bar, val in zip(bars, data):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(val), ha='center', fontweight='bold')

    return_acf1 = [all_memory[t]['return']['acf_lag1'] for t in BANKS]
    abs_acf1 = [all_memory[t]['abs_return']['acf_lag1'] for t in BANKS]

    x = np.arange(len(BANKS))
    w = 0.35
    axes[2].bar(x - w/2, return_acf1, w, label='Return', color='#3498db', alpha=0.8)
    axes[2].bar(x + w/2, abs_acf1, w, label='|Return|', color='#e74c3c', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(BANKS)
    axes[2].set_ylabel('ACF(1)')
    axes[2].set_title('ACF at Lag 1')
    axes[2].legend()
    axes[2].axhline(y=0, color='gray', linewidth=0.5)

    plt.suptitle('Memory Summary — Cross-Bank Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'memory_summary.png')
    plt.close()
    print(f"  Saved: memory_summary.png")


# ============================================================
# 6. SAVE RESULTS
# ============================================================
def save_results(all_memory):
    """Save memory analysis results to CSV."""
    rows = []
    for ticker in BANKS:
        for target in ['return', 'abs_return']:
            m = all_memory[ticker][target]
            halflife = compute_halflife(m['acf_vals'])
            rows.append({
                'bank': ticker,
                'series': target,
                'acf_lag1': m['acf_lag1'],
                'acf_lag5': m['acf_lag5'],
                'acf_lag10': m['acf_lag10'],
                'acf_lag20': m['acf_lag20'],
                'n_significant_lags': m['n_significant_lags'],
                'first_insignificant_lag': m['first_insignificant_lag'],
                'halflife_lag': halflife,
                'ljungbox_p_lag5': m['ljungbox'].loc[5, 'lb_pvalue'],
                'ljungbox_p_lag10': m['ljungbox'].loc[10, 'lb_pvalue'],
                'ljungbox_p_lag20': m['ljungbox'].loc[20, 'lb_pvalue'],
            })

    df_results = pd.DataFrame(rows)
    df_results.to_csv(OUTPUT_DIR / 'memory_analysis_results.csv', index=False)
    print(f"  Saved: memory_analysis_results.csv")
    return df_results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("LANG KINH 3: TFT + ACF — Market Memory Analysis")
    print("Cau hoi: Bao xa tri nho thi truong keo dai?")
    print("=" * 60)

    all_memory = {}
    all_attention = {}

    for ticker in BANKS:
        print(f"\n{'#'*50}")
        print(f"# {ticker}")
        print(f"{'#'*50}")

        df = load_data(ticker)
        print(f"  Data: {len(df)} rows, {df['date'].min().date()} -> {df['date'].max().date()}")

        # Part A: ACF/PACF plots
        plot_acf_pacf_single(df, ticker)

        # Memory analysis
        memory = analyze_memory(df, ticker)
        all_memory[ticker] = memory

        # Part B: TFT Attention
        attention = train_tft_and_get_attention(df, ticker)
        all_attention[ticker] = attention

    # ===== CROSS-BANK =====
    print(f"\n{'='*60}")
    print("CROSS-BANK ANALYSIS")
    print(f"{'='*60}")

    plot_crossbank_acf(all_memory)
    plot_memory_summary(all_memory)
    df_results = save_results(all_memory)

    # Part B: TFT attention plots
    plot_tft_attention(all_attention)

    # Part C: ACF vs TFT Attention
    plot_acf_vs_attention(all_memory, all_attention)

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*60}")
    print("KET QUA CHINH")
    print(f"{'='*60}")

    print("\n1. Return (gia tang/giam) — Co phai random walk?")
    print(f"   {'Bank':<6} {'ACF(1)':<10} {'Sig.lags':<12} {'Ljung-Box(10)':<18} {'Ket luan'}")
    print(f"   {'-'*65}")
    for ticker in BANKS:
        m = all_memory[ticker]['return']
        lb_p = m['ljungbox'].loc[10, 'lb_pvalue']
        conclusion = "Gan random walk" if m['n_significant_lags'] < 5 else "Co memory"
        print(f"   {ticker:<6} {m['acf_lag1']:>+.4f}    {m['n_significant_lags']:<12} p={lb_p:.6f}      {conclusion}")

    print("\n2. |Return| (muc do bien dong) — Volatility clustering?")
    print(f"   {'Bank':<6} {'ACF(1)':<10} {'Sig.lags':<12} {'Half-life':<12} {'Ket luan'}")
    print(f"   {'-'*60}")
    for ticker in BANKS:
        m = all_memory[ticker]['abs_return']
        halflife = compute_halflife(m['acf_vals'])
        hl_str = f"{halflife} ngay" if halflife else "N/A"
        conclusion = "Volatility clustering MANH" if m['n_significant_lags'] > 10 else "Yeu"
        print(f"   {ticker:<6} {m['acf_lag1']:>+.4f}    {m['n_significant_lags']:<12} {hl_str:<12} {conclusion}")

    print("\n3. So sanh Return vs |Return|:")
    print("   Return: ACF gan 0 => gia tang/giam gan nhu NGAU NHIEN")
    print("   |Return|: ACF cao, decay cham => muc do bien dong CO QUY LUAT")
    print("   => Day la ly do GARCH hoat dong tot (bat duoc volatility clustering)")

    if any(att is not None for att in all_attention.values()):
        print("\n4. TFT Attention:")
        print("   Xem chart partB_tft_attention.png va partC_acf_vs_attention.png")

    print(f"\n\nAll output saved to: {OUTPUT_DIR}/")
    print("=" * 60)



if __name__ == "__main__":
    main()
