

https://mcwrarealtimehydrodata.com/export/flot/?method=sensorDetails&site_id=15530&device_id=1&data_start=2026-01-01&data_end=2026-01-02
https://wr.slocountywater.org/export/flot/?method=sensorDetails&site_id=11&device_id=1&data_start=2026-01-01&data_end=2026-01-02

sites = [
    (6208, 211), # Salinas River Lagoon
    (6193, 211), # Carmel River Lagoon
]

import datetime

timestamp_ms = 1767687031000

# Convert milliseconds to seconds
timestamp_sec = timestamp_ms / 1000

# Convert to datetime object
dt_object = datetime.datetime.fromtimestamp(timestamp_sec)

