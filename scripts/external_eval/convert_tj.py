from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd

PATH = "/Volumes/x10pro/estuary/water_data/raw/tj_download.csv"


df = pd.read_csv(PATH)
# Parse as naive local clock time first (Pacific time), then localize -> UTC.
dt = pd.to_datetime(df["DateTimeStamp"], format="%m/%d/%Y %H:%M", errors="coerce")
df["DateTimeStamp"] = dt.dt.tz_localize(timezone(timedelta(hours=-8))).dt.tz_convert("UTC")

df = df.rename(columns={"DateTimeStamp": "timestamp_utc", "Level": "height"})
df["region"] = 2163
df["source"] = "nerrs"
df["sensor_id"] = "TJROSWQ"

save_path = Path("/Volumes/x10pro/estuary/water_data/2163/nerrs.csv")
save_path.parent.mkdir(exist_ok=True, parents=True)
df.to_csv(save_path)
