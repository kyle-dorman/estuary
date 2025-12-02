from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# final image for PowerPoint
OUT_PNG = "/Users/kyledorman/data/estuary/stats_presentation/california_estuaries.png"

gdfs = [gpd.read_file(p) for p in Path("/Volumes/x10pro/estuary/ca_grids/").glob("*.geojson")]
assert all(gdf.crs == gdfs[0].crs for gdf in gdfs)
assert gdfs[0].crs is not None
gdf = gpd.GeoDataFrame(
    pd.concat(
        gdfs,
        ignore_index=True,
    ),
    geometry="geometry",
    crs=gdfs[0].crs,
).to_crs(4326)
# If polygons/lines are provided, use their centroids as site markers
pts = gdf.geometry.centroid

gdfs = [gpd.read_file(p) for p in Path("/Volumes/x10pro/estuary/skipped_grids/").glob("*.geojson")]
assert all(gdf.crs == gdfs[0].crs for gdf in gdfs)
assert gdfs[0].crs is not None
skip_gdf = gpd.GeoDataFrame(
    pd.concat(
        gdfs,
        ignore_index=True,
    ),
    geometry="geometry",
    crs=gdfs[0].crs,
).to_crs(4326)
# If polygons/lines are provided, use their centroids as site markers
skip_pts = skip_gdf.geometry.centroid

# Compute a padded extent around California points
minx, miny, maxx, maxy = gdf.total_bounds
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

# Add state boundaries for additional geographic context
ax.add_feature(
    cfeature.STATES.with_scale("10m"),
    edgecolor="#999999",
    linewidth=0.5,
    zorder=2,
)  # type: ignore


# Plot site markers
skip_scatter = ax.scatter(
    skip_pts.x,
    skip_pts.y,
    s=40,
    c="#e45756",  # warm red marker
    edgecolors="white",
    linewidths=0.9,
    zorder=4,
    transform=ccrs.PlateCarree(),
    label="Skipped estuaries",
)

# Plot site markers
valid_scatter = ax.scatter(
    pts.x,
    pts.y,
    s=40,
    c="#060489",  # skipped estuaries
    edgecolors="white",
    linewidths=0.9,
    zorder=5,
    transform=ccrs.PlateCarree(),
    label="Valid estuaries",
)


# Legend and subtle gridlines
legend = ax.legend(
    loc="lower left",
    frameon=True,
    framealpha=0.9,
    facecolor="white",
    edgecolor="#cccccc",
    title="Sites",
)

# Optional latitude/longitude gridlines for spatial reference
gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.3,
    color="gray",
    alpha=0.3,
    linestyle="--",
)
# Only show left/bottom labels to keep things clean
try:
    gl.top_labels = False
    gl.right_labels = False
except AttributeError:
    pass

# Title and frame
ax.set_title("California Estuary Sites", fontsize=16, weight="bold", pad=12)
# Keep a subtle outline instead of full axes
for spine in getattr(ax, "spines", {}).values():
    spine.set_visible(False)

# Save high-res PNG for PowerPoint
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved {OUT_PNG}")
