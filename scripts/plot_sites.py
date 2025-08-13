from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd

# final image for PowerPoint
OUT_PNG = "/Users/kyledorman/data/estuary/display/california_estuaries.png"

# Regions whose labels should be placed to the LEFT of the marker.
# Populate with exact strings from the GeoJSON "region" field (case-sensitive).
show_left = [
    "little_sur",
    "malibu_lagoon",
    "san_mateo_lagoon",
    "san_dieguito_lagoon",
]

# Regions whose labels should be placed BELOW the marker.
# Populate with exact strings from the GeoJSON "region" field (case-sensitive).
show_below = ["los_penasquitos_lagoon"]

# --- Load points ---
BASE = Path("/Users/kyledorman/data/estuary/skysat/tight_grids")
gdfs = [gpd.read_file(p) for p in BASE.glob("*.geojson")]
assert len(gdfs) > 0, f"No GeoJSONs found under {BASE}"
assert all(gdf.crs == gdfs[0].crs for gdf in gdfs), "All GeoJSONs must share the same CRS"

# Keep only geometry + region
gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry="geometry", crs=gdfs[0].crs)[
    ["geometry", "region"]
]

# Convert to WGS84 for cartopy plotting (lon/lat)
if gdf.crs is None:
    raise ValueError("Input GeoJSONs lack a CRS – please set one (e.g., EPSG:4326)")

gdf_ll = gdf.to_crs(4326)

# If polygons/lines are provided, use their centroids as site markers
pts = gdf_ll.geometry.centroid

# Compute a padded extent around California points
minx, miny, maxx, maxy = gdf_ll.total_bounds
pad_x = max(0.5, (maxx - minx) * 0.15)  # at least ~0.5° pad
pad_y = max(0.5, (maxy - miny) * 0.15)
extent = (minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y)

# --- Plot with Cartopy ---
fig = plt.figure(figsize=(12, 7), dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(extent, crs=ccrs.PlateCarree())  # type: ignore

# Ocean/Land background (Natural Earth, 10m)
ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dbe9ff", zorder=0)  # type: ignore
ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f6f6f4", edgecolor="none", zorder=1)  # type: ignore
ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#4a4a4a", linewidth=0.6, zorder=2)  # type: ignore
ax.add_feature(  # type: ignore
    cfeature.LAKES.with_scale("10m"),
    facecolor="#e8f2ff",
    edgecolor="#8aa6c1",
    linewidth=0.3,
    zorder=2,
)

# Plot site markers
ax.scatter(
    pts.x,
    pts.y,
    s=54,
    c="#e45756",  # warm red marker
    edgecolors="white",
    linewidths=0.9,
    zorder=5,
    transform=ccrs.PlateCarree(),
)

# Label styling
texts = []
# dynamic nudge roughly ~0.5% of span
dx = 0.01 * (extent[1] - extent[0])
dy = 0.006 * (extent[3] - extent[2])
for (x, y), region_name in zip(pts.apply(lambda p: (p.x, p.y)), gdf_ll["region"], strict=False):
    label = str(region_name).replace("_", " ").title()
    to_left = str(region_name) in show_left
    below = str(region_name) in show_below

    x_off = -dx if to_left else dx
    y_off = -dy if below else dy
    ha = "right" if to_left else "left"
    va = "top" if below else "bottom"

    t = ax.text(
        x + x_off,
        y + y_off,
        label,
        fontsize=10,
        weight="bold",
        color="#16365d",
        ha=ha,
        va=va,
        transform=ccrs.PlateCarree(),
        zorder=6,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white", alpha=0.9)],
    )
    texts.append(t)

# Title and frame
ax.set_title("California Estuary Sites", fontsize=16, weight="bold", pad=12)
# Keep a subtle outline instead of full axes
for spine in getattr(ax, "spines", {}).values():
    spine.set_visible(False)

# Save high-res PNG for PowerPoint
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved {OUT_PNG}")
