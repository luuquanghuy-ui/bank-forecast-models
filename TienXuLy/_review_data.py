import pandas as pd
import os

print("=" * 70)
print("FULL DATA REVIEW")
print("=" * 70)

# LK1
print("\n### LK1: XGBoost + SHAP ###\n")
for b in ['BID','CTG','VCB']:
    df = pd.read_csv(f'langkinh1_xgboost_shap/{b}_feature_importance.csv')
    print(f'{b} top 5 features:')
    for _, row in df.head(5).iterrows():
        print(f"  {row['feature']:20s} SHAP={row['mean_abs_shap']:.6f}  rank={int(row['rank'])}")
    g = pd.read_csv(f'langkinh1_xgboost_shap/{b}_group_importance.csv')
    for _, row in g.iterrows():
        print(f"  [{row['group']:10s}] {row['percentage']:.1f}%")
    print()

# LK2
print("\n### LK2: NeuralProphet + Stats ###\n")
for b in ['BID','CTG','VCB']:
    df = pd.read_csv(f'langkinh2_neuralprophet_seasonality/{b}_calendar_tests.csv')
    print(f'{b}:')
    for _, row in df.iterrows():
        sig = "***" if row['p_value'] < 0.01 else ("**" if row['p_value'] < 0.05 else "")
        print(f"  {row['test']:25s} p={row['p_value']:.4f} {sig}")
    print()

# LK3
print("\n### LK3: ACF + TFT ###\n")
df = pd.read_csv('langkinh3_tft_memory/memory_analysis_results.csv')
print(df.to_string(index=False))

# Phase 2
print("\n\n### Phase 2: 4-fold results ###\n")
df = pd.read_csv('four_fold_all_targets/4fold_5models_summary.csv')
print(df.to_string(index=False))

# Per bank
print("\n\nPer bank detail:")
for b in ['BID','CTG','VCB']:
    path = f'four_fold_all_targets/{b}_4fold_5models.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n{b}:")
        print(df.to_string(index=False))
