import streamlit as st
import pandas as pd
import pydeck as pdk
from pathlib import Path

st.set_page_config(
    page_title="Supply Chain Map: 3PL → DC → Store",
    layout="wide",
)

# ---------- CONFIG ----------
DATA_FILE = "Store Delivery Schedule-Geo.csv"  # CSV in same folder as this script


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure numeric types
    num_cols = [
        "Store Latitude",
        "Store Longitude",
        "DC Latitude",
        "DC Longitude",
        "Shipper Latitude",
        "Shipper Longitude",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------- LOAD DATA ----------
path = Path(DATA_FILE)
if not path.exists():
    st.error(f"Could not find data file: {path.resolve()}")
    st.stop()

df = load_data(str(path))

st.title("3PL → DC → Store Network Map")
st.caption("Green lines: 3PL → DC · Red lines: DC → Store")

# ---------- FILTER UI ----------
# Line filters
col1, col2 = st.columns(2)
with col1:
    show_shipper_dc = st.checkbox("Show 3PL → DC/DSD routes", value=True)
with col2:
    show_dc_store = st.checkbox("Show DC → Store routes", value=True)

# Point filters
colp1, colp2, colp3 = st.columns(3)
with colp1:
    show_shipper_points = st.checkbox("Show 3PL points", value=True)
with colp2:
    show_dc_points = st.checkbox("Show DC/DSD points", value=True)
with colp3:
    show_store_points = st.checkbox("Show Store points", value=True)

st.markdown("---")

# ---------- PREP POINT DATA ----------
# Stores
stores = (
    df[["Store Name", "Store Address", "Store Latitude", "Store Longitude"]]
    .dropna(subset=["Store Latitude", "Store Longitude"])
    .drop_duplicates()
    .rename(
        columns={
            "Store Name": "name",
            "Store Address": "address",
            "Store Latitude": "lat",
            "Store Longitude": "lon",
        }
    )
)
stores["kind"] = "Store"

# DCs
dcs = (
    df[["DC Address", "DC Latitude", "DC Longitude"]]
    .dropna(subset=["DC Latitude", "DC Longitude"])
    .drop_duplicates()
    .rename(
        columns={
            "DC Address": "name",
            "DC Latitude": "lat",
            "DC Longitude": "lon",
        }
    )
)
dcs["address"] = dcs["name"]
dcs["kind"] = "DC"

# Shippers
shippers = (
    df[["Shipper Address", "Shipper Latitude", "Shipper Longitude"]]
    .dropna(subset=["Shipper Latitude", "Shipper Longitude"])
    .drop_duplicates()
    .rename(
        columns={
            "Shipper Address": "name",
            "Shipper Latitude": "lat",
            "Shipper Longitude": "lon",
        }
    )
)
shippers["address"] = shippers["name"]
shippers["kind"] = "Shipper"

points = pd.concat([stores, dcs, shippers], ignore_index=True)
points["coordinates"] = points[["lon", "lat"]].values.tolist()
points["tooltip"] = (
    points["kind"] + ": " + points["name"].fillna("") + "<br/>" + points["address"].fillna("")
)

# ---------- PREP LINE DATA ----------
# Shipper → DC (green)
shipper_dc = df[
    [
        "Shipper Address",
        "Shipper Latitude",
        "Shipper Longitude",
        "DC Address",
        "DC Latitude",
        "DC Longitude",
    ]
].dropna(subset=["Shipper Latitude", "Shipper Longitude", "DC Latitude", "DC Longitude"])

shipper_dc = shipper_dc.drop_duplicates()
shipper_dc["source"] = shipper_dc[["Shipper Longitude", "Shipper Latitude"]].values.tolist()
shipper_dc["target"] = shipper_dc[["DC Longitude", "DC Latitude"]].values.tolist()
shipper_dc["tooltip"] = (
    "Shipper → DC<br/>Shipper: "
    + shipper_dc["Shipper Address"].fillna("")
    + "<br/>DC: "
    + shipper_dc["DC Address"].fillna("")
)

# DC → Store (red)
dc_store = df[
    [
        "DC Address",
        "DC Latitude",
        "DC Longitude",
        "Store Name",
        "Store Address",
        "Store Latitude",
        "Store Longitude",
    ]
].dropna(subset=["Store Latitude", "Store Longitude", "DC Latitude", "DC Longitude"])

dc_store = dc_store.drop_duplicates()
dc_store["source"] = dc_store[["DC Longitude", "DC Latitude"]].values.tolist()
dc_store["target"] = dc_store[["Store Longitude", "Store Latitude"]].values.tolist()
dc_store["tooltip"] = (
    "DC → Store<br/>DC: "
    + dc_store["DC Address"].fillna("")
    + "<br/>Store: "
    + dc_store["Store Name"].fillna("")
    + "<br/>"
    + dc_store["Store Address"].fillna("")
)

# ---------- VIEW STATE (this MUST come before pdk.Deck) ----------
if not points.empty:
    center_lat = points["lat"].mean()
    center_lon = points["lon"].mean()
else:
    center_lat, center_lon = 39.5, -98.35  # fallback center of US

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=5,
    pitch=0,
)

# ---------- LAYERS ----------
layers = []

# Shipper points (blue)
if show_shipper_points and not shippers.empty:
    shipper_points = points[points["kind"] == "Shipper"]
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=shipper_points,
            get_position="coordinates",
            get_radius=8,                  # in pixels now
            radius_units="pixels",         # <- key change
            radius_min_pixels=1,           # avoid too tiny points
            radius_max_pixels=5,           # avoid giant circles when zoomed in
            get_fill_color=[27, 90, 125],  # blue
            pickable=True,
        )
    )

# DC points (red)
if show_dc_points and not dcs.empty:
    dc_points = points[points["kind"] == "DC"]
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=dc_points,
            get_position="coordinates",
            get_radius=8,                  # in pixels now
            radius_units="pixels",         # <- key change
            radius_min_pixels=1,           # avoid too tiny points
            radius_max_pixels=5,           # avoid giant circles when zoomed in
            get_fill_color=[96, 24, 53],  # red
            pickable=True,
        )
    )

# Store points (green)
if show_store_points and not stores.empty:
    store_points = points[points["kind"] == "Store"]
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=store_points,
            get_position="coordinates",
            get_radius=8,                  # in pixels now
            radius_units="pixels",         # <- key change
            radius_min_pixels=1,           # avoid too tiny points
            radius_max_pixels=5,           # avoid giant circles when zoomed in
            get_fill_color=[0, 167, 93],  # green
            pickable=True,
        )
    )

# Shipper → DC lines (green)
if show_shipper_dc and not shipper_dc.empty:
    layers.append(
        pdk.Layer(
            "LineLayer",
            data=shipper_dc,
            get_source_position="source",
            get_target_position="target",
            get_width=3,                   # thickness in pixels
            width_units="pixels",          # explicit
            width_min_pixels=1,
            width_max_pixels=2,
            get_color=[0, 167, 93],  # green
            pickable=True,
        )
    )

# DC → Store lines (yellow)
if show_dc_store and not dc_store.empty:
    layers.append(
        pdk.Layer(
            "LineLayer",
            data=dc_store,
            get_source_position="source",
            get_target_position="target",
            get_width=3,                   # thickness in pixels
            width_units="pixels",          # explicit
            width_min_pixels=1,
            width_max_pixels=2,
            get_color=[254, 207, 102],  # yellow
            pickable=True,
        )
    )

r = pdk.Deck(
    map_provider="carto",           # use Carto's free basemaps
    map_style="road",               # 'light', 'dark', 'road', 'satellite'
    initial_view_state=view_state,
    layers=layers,
    tooltip={
        "html": "{tooltip}",
        "style": {
            "backgroundColor": "rgba(0, 0, 0, 0.8)",
            "color": "white",
        },
    },
)

# ---------- RENDER ----------
st.pydeck_chart(r)

st.markdown(
    """
**Legend**

- **Blue dots** = 3PL  
- **Red dots** = DCs  
- **Green dots** = Stores  
- **Green lines** = 3PL → DC  
- **Yellow lines** = DC → Store
"""
)