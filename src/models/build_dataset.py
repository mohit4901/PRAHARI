import os
import pandas as pd
from pathlib import Path
from uni2ts.data.builder.simple import SimpleDatasetBuilder

def build_moirai_dataset():
    data_path = Path("data/processed/omni_processed.csv")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=["Datetime"], index_col="Datetime")
    
    # We will build a dataset for uni2ts
    # uni2ts expects data in a specific storage format
    out_dir = Path("data/moirai_dataset")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Building uni2ts dataset at {out_dir}...")
    
    builder = SimpleDatasetBuilder(
        dataset="prahari_omni",
        storage_path=out_dir
    )
    
    # Add target (Pc5 Wave Power)
    target = df["Pc5_Wave_Power"].values
    
    builder.write(
        target=target,
        item_id="omni_wave_power",
        start_time=df.index[0]
    )
    print("Dataset built successfully!")

if __name__ == "__main__":
    build_moirai_dataset()
