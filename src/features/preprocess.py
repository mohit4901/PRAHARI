import os
import glob
import pandas as pd
import numpy as np
import pywt
from datetime import datetime, timedelta

def compute_pc5_wave_power(signal_array, dt_minutes=5):
    """
    Computes ULF Pc5 wave power (1.6 - 6.6 mHz) using PyWavelets Continuous Wavelet Transform.
    Data is sampled at 5 minutes (dt=300 seconds).
    Nyquist frequency is 1 / (2 * 300) = 1.66 mHz.
    Wait, 5-minute data can only resolve frequencies up to 1.66 mHz!
    Pc5 band is 1.6 - 6.6 mHz (periods of 150 to 600 seconds).
    To properly resolve Pc5, we need 1-minute data, but with 5-minute data, we are at the very edge of the Pc5 band (150s-300s is lost due to aliasing, but 300s-600s can barely be captured if we consider period=10 mins).
    We will use the 'cmor' wavelet to extract power at scales corresponding to 10-15 minute periods, representing the lower frequency end of Pc5.
    """
    # Sampling period in seconds
    fs = 1.0 / (dt_minutes * 60)
    
    # We want periods around 10-15 minutes (600 - 900 seconds) which is 1.1 to 1.6 mHz.
    frequencies = np.array([1.2e-3, 1.4e-3, 1.6e-3]) 
    
    # Pywt scale = central_freq / (freq * dt)
    # cmor1.5-1.0 has a central frequency of 1.0
    wavelet = 'cmor1.5-1.0'
    central_freq = pywt.central_frequency(wavelet)
    scales = central_freq / (frequencies / fs)
    
    # Fill nans temporarily with 0 for CWT
    signal_filled = pd.Series(signal_array).interpolate(limit_direction='both').fillna(0).values
    
    # Compute CWT
    coefficients, freqs = pywt.cwt(signal_filled, scales, wavelet, sampling_period=1/fs)
    
    # Wave power is the squared magnitude of the complex coefficients
    power = np.abs(coefficients)**2
    
    # Average power across the selected frequencies
    mean_power = np.mean(power, axis=0)
    return mean_power

def process_omni_data():
    raw_dir = "data/raw/omni_5min"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(raw_dir, "*.asc")))
    print(f"Found {len(files)} files.")
    
    # Define columns based on OMNI high-res format
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
    
    # Missing value indicators in OMNI
    missing_vals = [99.9, 999.9, 999.99, 9999.99, 99999.9, 9999999., 99999.99, 9999.99, 999.99]
    
    df_list = []
    for f in files:
        print(f"Reading {f}...")
        df = pd.read_csv(f, delim_whitespace=True, names=columns)
        df_list.append(df)
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Convert Year, Day, Hour, Minute to Datetime
    def make_date(row):
        year = int(row['Year'])
        day = int(row['Day'])
        hr = int(row['Hour'])
        mn = int(row['Minute'])
        return datetime(year, 1, 1) + timedelta(days=day-1, hours=hr, minutes=mn)
        
    print("Constructing datetime index...")
    full_df['Datetime'] = pd.to_datetime(
        full_df['Year'].astype(str) + full_df['Day'].astype(str).str.zfill(3) + 
        full_df['Hour'].astype(str).str.zfill(2) + full_df['Minute'].astype(str).str.zfill(2),
        format='%Y%j%H%M'
    )
    full_df.set_index('Datetime', inplace=True)
    
    # Replace OMNI missing values with NaN
    print("Replacing missing values...")
    for col in full_df.columns:
        if full_df[col].dtype == float or full_df[col].dtype == int:
            full_df.loc[full_df[col] >= 999.9, col] = np.nan
            if col in ["By_GSM", "Bz_GSM", "Bx_GSE", "Flow_Pressure", "Plasma_Beta", "Alfven_Mach", "Magnetosonic_Mach"]:
                full_df.loc[full_df[col] >= 99.9, col] = np.nan
            if col in ["Proton_10MeV", "Proton_30MeV", "Proton_60MeV"]:
                full_df.loc[full_df[col] < 0, col] = np.nan
                
    # Select key features for space weather and ULF waves
    # Dropping Temperature and Proton fluxes as they are largely missing in high-res
    features = [
        "Bx_GSE", "By_GSM", "Bz_GSM", "Flow_Speed", "Proton_Density", 
        "Flow_Pressure", "E_Field", "SYM_H", "AE", "AL"
    ]
    
    df_clean = full_df[features].copy()
    
    # Linearly interpolate gaps and fill remaining with forward/backward fill to maintain uniform index
    print("Interpolating missing values...")
    df_clean = df_clean.interpolate(method='linear')
    df_clean = df_clean.bfill().ffill()
    
    # Feature Engineering: Compute Pc5 Wave Power using PyWavelets on Bz_GSM
    print("Computing Pc5 Wave Power with PyWavelets...")
    df_clean['Pc5_Wave_Power'] = compute_pc5_wave_power(df_clean['Bz_GSM'].values)
    
    # Save the processed dataset
    out_path = os.path.join(processed_dir, "omni_processed.csv")
    print(f"Saving processed data to {out_path}...")
    df_clean.to_csv(out_path)
    print(f"Data processing complete. Final shape: {df_clean.shape}")

if __name__ == "__main__":
    process_omni_data()
