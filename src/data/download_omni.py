import os
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# NASA SPDF OMNI High-Res 5-minute data contains Wind solar wind + GOES >2 MeV electron flux
# Data format: CDF files per month
BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_5min/"
SAVE_DIR = "../../data/raw/omni_5min"

def download_month(year, month):
    # e.g., omni_hro2_5min_20130101_v01.cdf (The exact naming convention varies, usually they bundle it by month or day)
    # SPDF usually serves daily files for HRO2: omni_hro2_5min_20130101_v01.cdf
    # To avoid 4000+ requests, we will download the yearly ASCII or monthly CDFs.
    pass

def download_omni_yearly_ascii(year):
    # NASA SPDF provides yearly ASCII files for 5-minute data which are much easier to download!
    # URL: https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_minYYYY.asc
    url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_5min{year}.asc"
    save_path = os.path.join(SAVE_DIR, f"omni_5min_{year}.asc")
    
    if os.path.exists(save_path):
        print(f"[{year}] Already exists, skipping.")
        return
        
    print(f"[{year}] Downloading {url} ...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"[{year}] Successfully downloaded.")
        else:
            print(f"[{year}] Failed with status {response.status_code}")
    except Exception as e:
        print(f"[{year}] Error: {e}")

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    years = list(range(2013, 2025)) # 11+ years of data
    
    print(f"Starting highly-parallel download for {len(years)} years of OMNI Space Weather data (GOES + Wind)...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(download_omni_yearly_ascii, years)
        
    print("Download Phase Complete. Please proceed to parsing.")

if __name__ == "__main__":
    main()
