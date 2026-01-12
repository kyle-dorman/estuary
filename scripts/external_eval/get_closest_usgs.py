import json
import warnings

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")

ca_points = gpd.read_file("/Volumes/x10pro/estuary/geos/ca_data_w_empa_pmep.geojson")


SKIP = [
    13,
    22,
    29,
    23,
    28,
    25,
    31,
    45,
    46,
    36,
    48,
    55,
    13073,
    56,
    2138,
    13057,
    57,
    54,
    53,
    52,
    64,
    71,
    77,
    83,
    96,
    95,
    12103,
    14,
]


def get_usgs_sites(lat, lon, buffer_deg):
    """
    Query USGS NWIS Site Service for sites near (lat, lon).
    buffer_deg: search radius in degrees (~0.1 ≈ 10km at CA latitudes).
    params: USGS parameter codes (00010=water temp, 00065=gage height).
    """
    params = ["00010", "00060", "00065", "63160"]
    bbox = f"{lon - buffer_deg:0.3f},{lat - buffer_deg:0.3f},{lon + buffer_deg:0.3f},{lat + buffer_deg:0.3f}"
    url = (
        "https://waterservices.usgs.gov/nwis/iv/"
        f"?format=json&bBox={bbox}&parameterCd={','.join(params)}"
    )
    r = requests.get(url)
    r.raise_for_status()

    data = json.loads(r.text)["value"]["timeSeries"]
    df = pd.DataFrame(
        [
            {
                "siteName": d["sourceInfo"]["siteName"],
                "siteCode": d["sourceInfo"]["siteCode"][0]["value"],
                "latitude": d["sourceInfo"]["geoLocation"]["geogLocation"]["latitude"],
                "longitude": d["sourceInfo"]["geoLocation"]["geogLocation"]["longitude"],
            }
            for d in data
        ]
    )
    return df


def nearest_usgs_site(lat, lon, buffer_deg=0.05):
    sites = get_usgs_sites(lat, lon, buffer_deg=buffer_deg)
    if sites.empty:
        return None

    # Convert to GeoDataFrame
    sites["geometry"] = gpd.points_from_xy(sites.longitude, sites.latitude)
    gdf = gpd.GeoDataFrame(sites, geometry="geometry", crs="EPSG:4326")

    # Compute distances
    pt = Point(lon, lat)
    gdf["distance"] = gdf.geometry.distance(pt)
    return gdf.sort_values("distance").iloc[0]


ca_with_river = ca_points.copy()
ca_with_river["usgs_station_nm"] = None
ca_with_river["usgs_site_no"] = None
ca_with_river["usgs_site_latitude"] = None
ca_with_river["usgs_site_longitude"] = None
for idx, est_row in ca_with_river.iterrows():
    if est_row["Site code"] in SKIP:
        continue
    lat, lon = est_row.geometry.centroid.y, est_row.geometry.centroid.x
    site = nearest_usgs_site(lat, lon, buffer_deg=0.05)
    if site is not None:
        print(est_row["Estuary_Name"], site.siteName, est_row["Site code"])
        ca_with_river.loc[idx, "usgs_station_nm"] = site.siteName
        ca_with_river.loc[idx, "usgs_site_no"] = site.siteCode
        ca_with_river.loc[idx, "usgs_site_latitude"] = site.latitude
        ca_with_river.loc[idx, "usgs_site_longitude"] = site.longitude


ca_with_river.to_file("/Volumes/x10pro/estuary/geos/ca_data_w_empa_pmep_usgs.geojson")
