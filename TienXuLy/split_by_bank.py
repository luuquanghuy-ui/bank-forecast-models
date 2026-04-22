from pathlib import Path

import pandas as pd


BASE_DIR = Path(r"D:\labs 2\DOANPTDLKD")
SOURCE_FILE = BASE_DIR / "banks_master_dataset_final - banks_master_dataset_final.csv"


def main() -> None:
    df = pd.read_csv(SOURCE_FILE, parse_dates=["date"])
    output_files = []

    for ticker in sorted(df["ticker"].unique()):
        subset = df[df["ticker"] == ticker].sort_values("date").copy()
        output_path = BASE_DIR / f"banks_{ticker}_dataset.csv"
        subset.to_csv(output_path, index=False, encoding="utf-8-sig")
        output_files.append((ticker, output_path.name, len(subset), subset["date"].min().date(), subset["date"].max().date()))

    print(f"Source file: {SOURCE_FILE}")
    print()
    print("Created split files:")
    for ticker, name, rows, start, end in output_files:
        print(f"- {ticker}: {name} | rows={rows} | {start} -> {end}")


if __name__ == "__main__":
    main()
