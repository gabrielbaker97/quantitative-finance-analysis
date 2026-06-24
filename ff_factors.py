import pandas as pd
import numpy as np
import io
import re
import zipfile
from curl_cffi import requests

start_date = "1960-01-01"
end_date = "2024-12-31"

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

factors_ff_raw = pd.read_csv(io.StringIO(csv_text), index_col=0)

s = factors_ff_raw.index.astype(str)

if (s.str.len() == 8).all():  # daily: YYYYMMDD
    dt = pd.to_datetime(s, format="%Y%m%d")
elif (s.str.len() == 6).all():  # monthly: YYYYMM
    dt = pd.to_datetime(s + "01", format="%Y%m%d")
elif (s.str.len() == 4).all():  # annual: YYYY
    dt = pd.to_datetime(s + "0101", format="%Y%m%d")
    dt = dt.dt.to_period("A-DEC").dt.to_timestamp("end")
else:
    raise ValueError("Unknown date format in Fama–French index.")

factors_ff_raw = factors_ff_raw.set_index(dt)
factors_ff_raw.index.name = "date"

# start and end dates
if start_date:
    factors_ff_raw = factors_ff_raw[factors_ff_raw.index >= pd.to_datetime(start_date)]
if end_date:
    factors_ff_raw = factors_ff_raw[factors_ff_raw.index <= pd.to_datetime(end_date)]

factors_ff3_monthly = (factors_ff_raw
    .div(100)
    .reset_index(names="date")
    .rename(columns=str.lower)
    .rename(columns={"mkt-rf": "mkt_excess", "rf": "risk_free"})
    .replace({"-99.99": pd.NA, -99.99: pd.NA, -999: pd.NA})
)
factors_ff3_monthly