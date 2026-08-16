import pandas as pd
import numpy as np
import glob
import os

columns = [
    "Year", "Day", "Hour", "Minute", "ID_IMF", "ID_SW", "Num_IMF", "Num_SW", 
    "Percent_Interp", "Timeshift", "RMS_Timeshift", "RMS_Phase", "Time_Btwn_Obs", 
    "B_mag_avg", "Bx_GSE", "By_GSE", "Bz_GSE", "By_GSM", "Bz_GSM", 
    "RMS_SD_B", "RMS_SD_B_vector", "Flow_Speed", "Vx", "Vy", "Vz", 
    "Proton_Density", "Temperature", "Flow_Pressure", "E_Field", "Plasma_Beta", 
    "Alfven_Mach", "X_sc", "Y_sc", "Z_sc", "BS_X", "BS_Y", "BS_Z", 
    "AE", "AL", "AU", "SYM_D", "SYM_H", "ASY_D", "ASY_H", "PC_N", 
    "Magnetosonic_Mach", "Proton_10MeV", "Proton_30MeV", "Proton_60MeV"
]

files = sorted(glob.glob("data/raw/omni_5min/*.asc"))
print("Reading file:", files[-1])
df = pd.read_csv(files[-1], delim_whitespace=True, names=columns)

for col in df.columns:
    if df[col].dtype == float or df[col].dtype == int:
        df.loc[df[col] >= 999.9, col] = np.nan
        if col in ["By_GSM", "Bz_GSM", "Bx_GSE"]:
            df.loc[df[col] >= 99.9, col] = np.nan
        if col in ["Proton_10MeV", "Proton_30MeV", "Proton_60MeV"]:
            df.loc[df[col] < 0, col] = np.nan

features = [
    "Bx_GSE", "By_GSM", "Bz_GSM", "Flow_Speed", "Proton_Density", 
    "Temperature", "Flow_Pressure", "E_Field", "SYM_H", "AE", "AL",
    "Proton_10MeV"
]

print("Missing value counts for last year:")
for f in features:
    print(f"{f}: {df[f].isna().sum()} / {len(df)}")
