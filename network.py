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
DSD_COL = "DC or DSD"


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

# Normalize DC / DSD flag if present
has_dsd_col = DSD_COL in df.columns
if has_dsd_col:
    raw_flag = df[DSD_COL].fillna("").astype(str).str.strip().str.upper()
    # Treat anything not explicitly "DSD" as "DC"
    df[DSD_COL] = raw_flag.where(raw_flag.isin(["DC", "DSD"]), "DC")

    # DC rows: real DC network, DSD rows: direct 3PL → Store
    df_dc = df[df[DSD_COL] != "DSD"].copy()
    df_dsd = df[df[DSD_COL] == "DSD"].copy()
else:
    df_dc = df.copy()
    df_dsd = df.iloc[0:0].copy()  # empty frame with same columns

st.title("3PL → DC → Store Network Map")
st.caption(
    "Green lines: 3PL → DC · Yellow lines: DC → Store (DC) · Light blue lines: 3PL → DSD Store"
)

# ---------- FILTER UI ----------
# Line filters
if has_dsd_col:
    col1, col2, col3 = st.columns(3)
else:
    col1, col2 = st.columns(2)
    col3 = None

with col1:
    show_shipper_dc = st.checkbox("Show 3PL → DC routes", value=True)
with col2:
    show_dc_store = st.checkbox("Show DC → Store (DC) routes", value=True)
if has_dsd_col and col3 is not None:
    with col3:
        show_shipper_dsd = st.checkbox("Show 3PL → DSD Store routes", value=True)
else:
    show_shipper_dsd = False

# Point filters
if has_dsd_col:
    colp1, colp2, colp3, colp4 = st.columns(4)
else:
    colp1, colp2, colp3 = st.columns(3)
    colp4 = None

with colp1:
    show_shipper_points = st.checkbox("Show 3PL points", value=True)
with colp2:
    show_dc_points = st.checkbox("Show DC points", value=True)

if has_dsd_col:
    with colp3:
        show_dc_store_points = st.checkbox("Show DC Stores", value=True)
    with colp4:
        show_dsd_store_points = st.checkbox("Show DSD Stores", value=True)
else:
    with colp3:
        show_dc_store_points = st.checkbox("Show Store points", value=True)
    show_dsd_store_points = False

st.markdown("---")

# ---------- PREP STORE DATA ----------
store_cols = [
    "Store Name",
    "Store Address",
    "Store Latitude",
    "Store Longitude",
]
if has_dsd_col:
    store_cols.append(DSD_COL)

stores_base = (
    df[store_cols]
    .dropna(subset=["Store Latitude", "Store Longitude"])
    .copy()
)

if has_dsd_col:
    def collapse_delivery_model(series: pd.Series) -> str:
        vals = series.dropna().astype(str).str.strip().str.upper()
        if (vals == "DSD").any():
            return "DSD"
        return "DC"

    stores = (
        stores_base
        .groupby(
            ["Store Name", "Store Address", "Store Latitude", "Store Longitude"],
            as_index=False,
        )
        .agg({DSD_COL: collapse_delivery_model})
    )
else:
    stores = stores_base.drop_duplicates()

stores = stores.rename(
    columns={
        "Store Name": "name",
        "Store Address": "address",
        "Store Latitude": "lat",
        "Store Longitude": "lon",
    }
)
stores["kind"] = "Store"

if has_dsd_col:
    stores["delivery_model"] = (
        stores[DSD_COL]
        .fillna("DC")
        .astype(str)
        .str.strip()
        .str.upper()
        .where(lambda s: s.isin(["DC", "DSD"]), "DC")
    )

    def store_color(model: str):
        # DSD stores are pink, DC stores stay green
        return [246, 199, 206] if model == "DSD" else [0, 167, 93]

    stores["color"] = stores["delivery_model"].apply(store_color)
    stores["tooltip"] = (
        "Store (" + stores["delivery_model"] + "): "
        + stores["name"].fillna("")
        + "<br/>"
        + stores["address"].fillna("")
    )
else:
    stores["color"] = [0, 167, 93]
    stores["tooltip"] = (
        "Store: " + stores["name"].fillna("") + "<br/>" + stores["address"].fillna("")
    )

# ---------- PREP DC DATA ----------
if not df_dc.empty:
    dcs = (
        df_dc[["DC Address", "DC Latitude", "DC Longitude"]]
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

    # Remove DCs that are actually DSD stores (same address or same coordinates)
    if has_dsd_col:
        dsd_stores = stores[stores["delivery_model"] == "DSD"]
        dsd_coords = (
            dsd_stores[["lat", "lon"]]
            .dropna()
            .drop_duplicates()
        )
        dsd_addresses = dsd_stores["address"].dropna().unique()

        if not dsd_coords.empty:
            dcs = dcs.merge(
                dsd_coords.assign(_is_dsd_coord=True),
                on=["lat", "lon"],
                how="left",
            )
            dcs["_is_dsd_coord"] = dcs["_is_dsd_coord"].fillna(False)
        else:
            dcs["_is_dsd_coord"] = False

        dcs["_is_dsd_addr"] = (
            dcs["address"].isin(dsd_addresses) if len(dsd_addresses) > 0 else False
        )

        dcs = dcs[~(dcs["_is_dsd_coord"] | dcs["_is_dsd_addr"])].copy()
        dcs = dcs.drop(columns=["_is_dsd_coord", "_is_dsd_addr"])
    dcs["tooltip"] = "DC: " + dcs["name"].fillna("") + "<br/>" + dcs["address"].fillna("")
else:
    dcs = pd.DataFrame(columns=["name", "lat", "lon", "address", "kind", "tooltip"])

# ---------- PREP SHIPPER DATA ----------
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
shippers["tooltip"] = (
    "3PL: " + shippers["name"].fillna("") + "<br/>" + shippers["address"].fillna("")
)

# ---------- COMBINED POINTS (for centering map) ----------
points = pd.concat([stores, dcs, shippers], ignore_index=True)
if not points.empty:
    points["coordinates"] = points[["lon", "lat"]].values.tolist()

# ---------- PREP LINE DATA ----------
# Shipper → DC (green), only for DC-based rows
if not df_dc.empty:
    shipper_dc = df_dc[
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
else:
    shipper_dc = pd.DataFrame()

# DC → Store (yellow), only for DC-based rows
if not df_dc.empty:
    dc_store = df_dc[
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
else:
    dc_store = pd.DataFrame()

# Shipper → DSD Store (light blue), only for DSD rows
if has_dsd_col and not df_dsd.empty:
    shipper_dsd = df_dsd[
        [
            "Shipper Address",
            "Shipper Latitude",
            "Shipper Longitude",
            "Store Name",
            "Store Address",
            "Store Latitude",
            "Store Longitude",
        ]
    ].dropna(subset=["Shipper Latitude", "Shipper Longitude", "Store Latitude", "Store Longitude"])

    shipper_dsd = shipper_dsd.drop_duplicates()
    shipper_dsd["source"] = shipper_dsd[["Shipper Longitude", "Shipper Latitude"]].values.tolist()
    shipper_dsd["target"] = shipper_dsd[["Store Longitude", "Store Latitude"]].values.tolist()
    shipper_dsd["tooltip"] = (
        "Shipper → DSD Store<br/>Shipper: "
        + shipper_dsd["Shipper Address"].fillna("")
        + "<br/>Store: "
        + shipper_dsd["Store Name"].fillna("")
        + "<br/>"
        + shipper_dsd["Store Address"].fillna("")
    )
else:
    shipper_dsd = pd.DataFrame()

# ---------- VIEW STATE ----------
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
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=shippers,
            get_position=["lon", "lat"],
            get_radius=8,                  # in pixels
            radius_units="pixels",
            radius_min_pixels=1,
            radius_max_pixels=5,
            get_fill_color=[27, 90, 125],  # blue
            pickable=True,
        )
    )

# DC points (red)
if show_dc_points and not dcs.empty:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=dcs,
            get_position=["lon", "lat"],
            get_radius=8,                  # in pixels
            radius_units="pixels",
            radius_min_pixels=1,
            radius_max_pixels=5,
            get_fill_color=[96, 24, 53],   # red
            pickable=True,
        )
    )

# Store points - DC stores (green) and DSD stores (pink)
if has_dsd_col:
    if show_dc_store_points:
        dc_store_points = stores[stores["delivery_model"] == "DC"]
        if not dc_store_points.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=dc_store_points,
                    get_position=["lon", "lat"],
                    get_radius=8,
                    radius_units="pixels",
                    radius_min_pixels=1,
                    radius_max_pixels=5,
                    get_fill_color="color",
                    pickable=True,
                )
            )
    if show_dsd_store_points:
        dsd_store_points = stores[stores["delivery_model"] == "DSD"]
        if not dsd_store_points.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=dsd_store_points,
                    get_position=["lon", "lat"],
                    get_radius=8,
                    radius_units="pixels",
                    radius_min_pixels=1,
                    radius_max_pixels=5,
                    get_fill_color="color",
                    pickable=True,
                )
            )
else:
    if show_dc_store_points and not stores.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=stores,
                get_position=["lon", "lat"],
                get_radius=8,
                radius_units="pixels",
                radius_min_pixels=1,
                radius_max_pixels=5,
                get_fill_color="color",
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
            width_units="pixels",
            width_min_pixels=1,
            width_max_pixels=2,
            get_color=[0, 167, 93],        # green
            pickable=True,
        )
    )

# DC → Store lines (yellow) for DC stores only
if show_dc_store and not dc_store.empty:
    layers.append(
        pdk.Layer(
            "LineLayer",
            data=dc_store,
            get_source_position="source",
            get_target_position="target",
            get_width=3,                   # thickness in pixels
            width_units="pixels",
            width_min_pixels=1,
            width_max_pixels=2,
            get_color=[254, 207, 102],     # yellow
            pickable=True,
        )
    )

# Shipper → DSD Store lines (Blue)
if show_shipper_dsd and not shipper_dsd.empty:
    layers.append(
        pdk.Layer(
            "LineLayer",
            data=shipper_dsd,
            get_source_position="source",
            get_target_position="target",
            get_width=3,                   # thickness in pixels
            width_units="pixels",
            width_min_pixels=1,
            width_max_pixels=2,
            get_color=[27, 90, 125],     # light blue
            pickable=True,
        )
    )

r = pdk.Deck(
    map_provider="carto",
    map_style="road",
    initial_view_state=view_state,
    layers=layers,
    tooltip={
        "html": "{tooltip}",
        "style": {
            "backgroundColor": "rgba(0, 0, 0, 0.8)",
            "color": "white",
        },
    },
    # remove height here – it's only for Jupyter
)

# ---------- RENDER ----------
st.pydeck_chart(
    r,
    use_container_width=True,  # or width="stretch" on newer Streamlit
    height=700,                # <- this is what controls the map height
)

st.markdown(
    """
**Legend**

- **Blue dots** = 3PL  
- **Red dots** = DCs  
- **Green dots** = DC Stores  
- **Pink dots** = DSD Stores  
- **Green lines** = 3PL → DC  
- **Yellow lines** = DC → Store (DC)  
- **Blue lines** = 3PL → DSD Store
"""
)