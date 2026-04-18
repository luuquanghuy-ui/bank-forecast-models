import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANKS = ['BID', 'CTG', 'VCB']
BANK_FILES = {t: f"d:/labs 2/DOANPTDLKD/TienXuLy/banks_{t}_dataset.csv" for t in BANKS}
OUTPUT_DIR = "d:/labs 2/DOANPTDLKD/thesis_charts"

# ============================================================
# 1. Histogram of log_return per bank
# ============================================================
print("Creating ch3_histogram_log_return.png...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, ticker in enumerate(BANKS):
    df = pd.read_csv(BANK_FILES[ticker])
    returns = df['log_return'].dropna()

    axes[i].hist(returns, bins=50, density=True, alpha=0.7, color=['#2E86AB','#A23B72','#F18F01'][i])
    axes[i].set_title(f'{ticker}', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('Log Return')
    axes[i].set_ylabel('Density')
    axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # Normal overlay
    from scipy import stats
    x = np.linspace(returns.min(), returns.max(), 100)
    axes[i].plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 'k--', alpha=0.5, label='Normal')
    axes[i].legend(fontsize=8)

plt.suptitle('Histogram of Daily Log Returns (vs Normal Distribution)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ch3_histogram_log_return.png', dpi=150)
plt.close()
print(f"  Saved: ch3_histogram_log_return.png")

# ============================================================
# 2. Boxplot of |log_return| per bank
# ============================================================
print("Creating ch3_boxplot_abs_return.png...")
fig, ax = plt.subplots(figsize=(8, 5))

data = []
labels = []
for ticker in BANKS:
    df = pd.read_csv(BANK_FILES[ticker])
    abs_ret = np.abs(df['log_return'].dropna())
    data.append(abs_ret)
    labels.append(ticker)

bp = ax.boxplot(data, labels=labels, patch_artist=True)
colors = ['#2E86AB', '#A23B72', '#F18F01']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('|Log Return|')
ax.set_title('Boxplot of Absolute Daily Returns by Bank', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ch3_boxplot_abs_return.png', dpi=150)
plt.close()
print(f"  Saved: ch3_boxplot_abs_return.png")

# ============================================================
# 3. Close price line chart - all 3 banks over time
# ============================================================
print("Creating ch3_close_price_chart.png...")
fig, ax = plt.subplots(figsize=(14, 6))

colors = {'BID': '#2E86AB', 'CTG': '#A23B72', 'VCB': '#F18F01'}
for ticker in BANKS:
    df = pd.read_csv(BANK_FILES[ticker])
    ax.plot(pd.to_datetime(df['date']), df['close'], label=ticker, color=colors[ticker], linewidth=0.8, alpha=0.85)

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Close Price (VND)', fontsize=11)
ax.set_title('Giá Đóng Cửa Ba Mã Ngân Hàng (2016-2026)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Add event annotations
ax.axvline(pd.to_datetime('2020-03-15'), color='red', linestyle='--', alpha=0.5, label='COVID Crash')
ax.text(pd.to_datetime('2020-03-15'), ax.get_ylim()[1]*0.95, 'COVID\nCrash', fontsize=8, color='red', ha='center')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ch3_close_price_chart.png', dpi=150)
plt.close()
print(f"  Saved: ch3_close_price_chart.png")

# ============================================================
# 4. Volume bar chart - all 3 banks over time
# ============================================================
print("Creating ch3_volume_chart.png...")
fig, ax = plt.subplots(figsize=(14, 6))

# Sample every 20th day to avoid overcrowding
for ticker in BANKS:
    df = pd.read_csv(BANK_FILES[ticker])
    dates = pd.to_datetime(df['date'])
    # Downsample for better visualization
    step = max(1, len(df) // 200)
    ax.bar(dates[::step], df['volume'].values[::step], label=ticker, color=colors[ticker], alpha=0.6, width=3)

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Volume (Million Shares)', fontsize=11)
ax.set_title('Khối Lượng Giao Dịch Ba Mã Ngân Hàng (2016-2026)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ch3_volume_chart.png', dpi=150)
plt.close()
print(f"  Saved: ch3_volume_chart.png")

print("\nDone!")
