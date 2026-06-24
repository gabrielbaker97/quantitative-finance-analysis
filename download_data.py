import os
from fredapi import Fred
import tidyfinance as tf
import pandas_datareader as pdr
import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np
from tqdm import tqdm 
import zipfile
import requests
from dotenv import load_dotenv

start_date = "1960-02-01"
end_date = "2024-12-01"

# ── Fama French Data ──────────────────────────────────────────────────
dataset = "F-F_Research_Data_Factors"
base_url = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
url = f"{base_url}{dataset}_CSV.zip"

resp = requests.get(url)
resp.raise_for_status()

with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
    file_name = zf.namelist()[0]  # Ken French ZIPs contain one file
    raw_text = zf.read(file_name).decode("latin1")

chunks = raw_text.split("\r\n\r\n")
table_text = max(chunks, key=len)

match = re.search(r"^\s*,", table_text, flags=re.M)
start = match.start()
csv_text = "Date" + table_text[start:]

ff_raw = pd.read_csv(io.StringIO(csv_text), index_col=0)

s = ff_raw.index.astype(str)

if (s.str.len() == 8).all():  # daily: YYYYMMDD
    dt = pd.to_datetime(s, format="%Y%m%d")
elif (s.str.len() == 6).all():  # monthly: YYYYMM
    dt = pd.to_datetime(s + "01", format="%Y%m%d")
elif (s.str.len() == 4).all():  # annual: YYYY
    dt = pd.to_datetime(s + "0101", format="%Y%m%d")
    dt = dt.dt.to_period("A-DEC").dt.to_timestamp("end")
else:
    raise ValueError("Unknown date format in Fama–French index.")

ff_raw = ff_raw.set_index(dt)
ff_raw.index.name = "date"

# start and end dates
if start_date:
    ff_raw = ff_raw[ff_raw.index >= pd.to_datetime(start_date)]
if end_date:
    ff_raw = ff_raw[ff_raw.index <= pd.to_datetime(end_date)]

ff3_monthly = (ff_raw
    .div(100)
    .reset_index(names="date")
    .rename(columns=str.lower)
    .rename(columns={"mkt-rf": "Mkt-RF", "smb" : "SMB", "hml": "HML", "rf": "RF"})
    .replace({"-99.99": pd.NA, -99.99: pd.NA, -999: pd.NA})
)
ff3_monthly


# ── Macro data from FRED ──────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY")

if FRED_API_KEY is None:
    raise ValueError(
        "Please set the FRED_API_KEY in environment variable"
    )

fred = Fred(api_key=FRED_API_KEY)

fred_series = {
    "DGS10": "treasury_10y",
    "DGS2":  "treasury_2y",
    "CPIAUCNS": "cpi",
    "UNRATE": "unemployment",
    "INDPRO": "industrial_prod",
}

macro_frames = []
for series, name in fred_series.items():
    df = (
        fred.get_series(series, observation_start=start_date, observation_end=end_date)
        .rename(name)
        .to_frame()
    )
    macro_frames.append(df)

macro = (
    pd.concat(macro_frames, axis=1)
    .reset_index()
    .rename(columns={"index": "date"})
)

macro["term_spread"] = macro["treasury_10y"] - macro["treasury_2y"]
print(macro.head())

(
    macro.isna()
    .sum()
    .rename("n_missing")
    .to_frame()
    .sort_values("n_missing", ascending=False)
)

# ── Save data ───────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
ff3_monthly.to_parquet("data/ff3_monthly.parquet", index=False)
macro.to_parquet("data/macro.parquet", index=False)



