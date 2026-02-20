import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="NYC Taxi Trip Dashboard",
    layout="wide"
)

st.title("🚕 NYC Taxi Trip Analysis Dashboard")
st.markdown("""
This dashboard explores NYC yellow taxi trip data.
Use the filters to explore specific dates, hours, and payment types.
""")

# -----------------------------
# Sidebar filters
# -----------------------------
raw_path = "data/raw"
zone_file = os.path.join(raw_path, "taxi_zone_lookup.csv")

# Load zones (small file, safe to load entirely)
df_zones = pd.read_csv(zone_file)

# Date range
min_date = pd.to_datetime("2024-01-01")
max_date = pd.to_datetime("2025-12-31")
date_range = st.sidebar.date_input(
    "Select Date Range", [min_date, max_date],
    min_value=min_date, max_value=max_date
)

# Hour range
hour_range = st.sidebar.slider("Select Hour Range", 0, 23, (0, 23))

# Payment type multiselect (we'll read unique values lazily later)
payment_types = st.sidebar.multiselect("Select Payment Types", ["1","2","3","4"], default=["1","2","3","4"])

# -----------------------------
# Lazy load data from DuckDB
# -----------------------------
# Parquet files pattern (all 2024-2025)
trip_files_pattern = os.path.join(raw_path, "yellow_tripdata_2024-*.parquet")

@st.cache_data
def load_filtered_data(start_date, end_date, hour_range, payment_types):
    # Build DuckDB query
    query = f"""
    SELECT *, 
           DATE(tpep_pickup_datetime) AS pickup_date,
           EXTRACT(hour FROM tpep_pickup_datetime) AS pickup_hour
    FROM read_parquet('{trip_files_pattern}')
    WHERE tpep_pickup_datetime BETWEEN '{start_date}' AND '{end_date}'
      AND EXTRACT(hour FROM tpep_pickup_datetime) BETWEEN {hour_range[0]} AND {hour_range[1]}
      AND payment_type IN ({','.join(str(p) for p in payment_types)})
    """
    df = duckdb.query(query).df()
    
    # Merge pickup and dropoff zones
    df = df.merge(df_zones[["LocationID","Zone"]], left_on="PULocationID", right_on="LocationID", how="left").rename(columns={"Zone":"PU_Zone"})
    df = df.merge(df_zones[["LocationID","Zone"]], left_on="DOLocationID", right_on="LocationID", how="left").rename(columns={"Zone":"DO_Zone"})
    
    # Feature engineering
    df["trip_duration_minutes"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    df["revenue"] = df["fare_amount"]
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.day_name()
    
    return df

# Load only filtered data
filtered = load_filtered_data(date_range[0], date_range[1], hour_range, payment_types)

# -----------------------------
# Key Metrics
# -----------------------------
st.subheader("Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trips", f"{len(filtered):,}")
col2.metric("Average Fare", f"${filtered['fare_amount'].mean():.2f}")
col3.metric("Total Revenue", f"${filtered['revenue'].sum():,.2f}")
col4.metric("Avg Distance (mi)", f"{filtered['trip_distance'].mean():.2f}")
col5.metric("Avg Duration (min)", f"{filtered['trip_duration_minutes'].mean():.2f}")

# -----------------------------
# Top 10 Pickup Zones
# -----------------------------
st.subheader("Top 10 Pickup Zones by Trip Count")
top_zones = filtered["PU_Zone"].value_counts().head(10).reset_index()
top_zones.columns = ["Zone", "Trips"]
fig1 = px.bar(top_zones, x="Zone", y="Trips", color="Trips", title="Top 10 Pickup Zones")
st.plotly_chart(fig1, width="stretch")

# -----------------------------
# Average Fare by Hour
# -----------------------------
st.subheader("Average Fare by Hour of Day")
hourly_fare = filtered.groupby("pickup_hour")["fare_amount"].mean().reset_index()
fig2 = px.line(hourly_fare, x="pickup_hour", y="fare_amount", markers=True, title="Average Fare by Hour")
st.plotly_chart(fig2, width="stretch")

# -----------------------------
# Trip Distance Histogram
# -----------------------------
st.subheader("Distribution of Trip Distances")
fig3 = px.histogram(filtered, x="trip_distance", nbins=40, title="Trip Distance Distribution")
st.plotly_chart(fig3, width="stretch")

# -----------------------------
# Payment Type Pie Chart
# -----------------------------
st.subheader("Payment Type Breakdown")
pay_counts = filtered["payment_type"].value_counts().reset_index()
pay_counts.columns = ["Payment Type","Count"]
fig4 = px.pie(pay_counts, names="Payment Type", values="Count", title="Payment Types")
st.plotly_chart(fig4, width="stretch")

# -----------------------------
# Heatmap: Trips by Day and Hour
# -----------------------------
st.subheader("Trips by Day of Week and Hour")
heatmap_data = filtered.groupby(["pickup_day_of_week","pickup_hour"]).size().reset_index(name="Trips")
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
heatmap_data["pickup_day_of_week"] = pd.Categorical(heatmap_data["pickup_day_of_week"], categories=day_order, ordered=True)
fig5 = px.density_heatmap(heatmap_data, x="pickup_hour", y="pickup_day_of_week", z="Trips", title="Trips by Day and Hour")
st.plotly_chart(fig5, width="stretch")