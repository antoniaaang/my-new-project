# %%
import xarray as xr

def main() -> None:
    print("Hello from my-new-project!")

# %%
print("hello world")
print("hi this is another line")
print("this is my third line that I am changing")
# %%
print("this is another commit wahoo")
print("okay this is the last time that I'm going to commit")
# %%
# %%
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

# %%
sms_dir = pathlib.Path("/work/mh0010/m300575/sms")
ncfiles = list((sms_dir / "1974_08_05_217/vis").glob("sms01.1974.*.A.vis.nc"))
ncfile = ncfiles[2]

ds = xr.open_dataset(ncfile, chunks={"xc": -1, "yc": -1})
ds = ds.load().isel(xc=slice(0, -1, 10), yc=slice(0, -1, 10))
ds.to_zarr(ncfile.with_suffix(".zarr").name, mode="w")
# %%
ds = xr.open_dataset(ncfile.with_suffix(".zarr").name)
ds.isel(xc=750).lat.plot()
# %%
ds.isel(xc=750, yc = 750)
# %%

# %%

# Look into distortion and movement of center point
# Calibrate so that all files can overlap 
# Understand the weird wobbling
# Plot a histogram of latitude and longitude

ncfiles

# %%
lat = ds["lat"].values
lon = ds["lon"].values

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(lat[np.isfinite(lat)], bins=100)
axes[0].set_title("Latitude Distribution")
axes[0].set_xlabel("Latitude (degrees)")
axes[0].set_ylabel("Pixel Count")

axes[1].hist(lon[np.isfinite(lon)], bins=100)
axes[1].set_title("Longitude Distribution")
axes[1].set_xlabel("Longitude (degrees)")
axes[1].set_ylabel("Pixel Count")

plt.tight_layout()
plt.show()
# %%
# How geographic coordinates vary across the image
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ds["lat"].plot(ax=axes[0], cmap="viridis")
axes[0].set_title("Latitude Across Image")

ds["lon"].plot(ax=axes[1], cmap="viridis")
axes[1].set_title("Longitude Across Image")

plt.tight_layout()
plt.show()
# %%
lat = ds["lat"].values
lon = ds["lon"].values

plt.figure(figsize=(8, 8))

step = max(1, min(lat.shape) // 20)

for i in range(0, lat.shape[0], step):
    plt.plot(lon[i, :], lat[i, :], "b-", alpha=0.4)

for j in range(0, lat.shape[1], step):
    plt.plot(lon[:, j], lat[:, j], "r-", alpha=0.4)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("SMS-1 Geographic Coordinate Grid")
plt.axis("equal")
plt.show()
# %%

# Plotting how the center latitude and longitude change over time 
center_lats = []
center_lons = []
filenames = []

for file in ncfiles:
    with xr.open_dataset(file) as ds_file:

        center_x = ds_file.sizes["xc"] // 2
        center_y = ds_file.sizes["yc"] // 2

        center = ds_file.isel(xc=center_x, yc=center_y)

        center_lats.append(float(center["lat"].values))
        center_lons.append(float(center["lon"].values))
        filenames.append(file.name)

center_lats = np.array(center_lats)
center_lons = np.array(center_lons)
# %%
plt.figure(figsize=(8, 6))

plt.plot(center_lons, center_lats, "o-", markersize=4)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("SMS-1 Image Center Movement")
plt.grid(True)
plt.axis("equal")

plt.show()
# %%
# Seeing how the distance from the first image changes over time
# Earth radius in km
R = 6371.0  

lat0 = np.radians(center_lats[0])
lon0 = np.radians(center_lons[0])

# loading all of the latitudes and longitudes
lat_rad = np.radians(center_lats)
lon_rad = np.unwrap(np.radians(center_lons))

dx = R * np.cos(lat0) * (lon_rad - lon0)
dy = R * (lat_rad - lat0)

# actual distance from the first image
displacement = np.sqrt(dx**2 + dy**2)

plt.figure(figsize=(10, 4))
plt.plot(displacement, "o-")

plt.xlabel("Image Index")
plt.ylabel("Displacement (km)")
plt.title("SMS-1 Center Displacement Relative to First Image")
plt.grid(True)
plt.show()

print("Maximum displacement:", np.max(displacement), "km")
# %%

# Ideas: use griddata() function
# Current problem: when the lat and lon are changing, the same pixel index may correspond to a different geographic location in each image

# If we overlay the images using pixel indices, the coastlines might appear to move

# %%
# Check dimensions and coordinate ranges for all files
records = []

for file in ncfiles:
    print("first check")
    with xr.open_dataset(file) as ds:
        print("opened file")
        lat = ds["lat"].values
        lon = ds["lon"].values

        records.append({
            "file": file.name,
            "xc": ds.sizes["xc"],
            "yc": ds.sizes["yc"],
            "lat_min": np.nanmin(lat),
            "lat_max": np.nanmax(lat),
            "lon_min": np.nanmin(lon),
            "lon_max": np.nanmax(lon),
        })
        print("appended to record")

df = pd.DataFrame(records)

print(df.to_string(index=False))
# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].plot(df["xc"], "o-")
axes[0].set_ylabel("xc pixels")
axes[0].set_title("Image Dimensions Across Files")

axes[1].plot(df["yc"], "o-")
axes[1].set_ylabel("yc pixels")
axes[1].set_xlabel("File Index")

plt.tight_layout()
plt.show()
# %%
# Plot center of latitude relative to pixel number
# fourier transform and signal processing