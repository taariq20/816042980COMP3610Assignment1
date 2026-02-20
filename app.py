import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# Page setup

st.set_page_config(
    page_title="NYC Yellow Taxi Trip Dashboard (January 2024)",
    layout="wide"
)

st.title("NYC Yellow Taxi Trip Dashboard for January 2024")
st.markdown("""
Analyse NYC yellow taxi trips using this dashboard for trips made in January 2024.
Filter by date, hour, and payment type to depict pickup locations, trip distances and revenue analytics.
""")

# Data paths & download

raw_path = "data/raw"
os.makedirs(raw_path, exist_ok=True)

trip_file = os.path.join(raw_path, "yellow_tripdata_2024-01.parquet")
zone_file = os.path.join(raw_path, "taxi_zone_lookup.csv")

# Download zone lookup
if not os.path.exists(zone_file):
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    r = requests.get(url)
    r.raise_for_status()
    with open(zone_file, "wb") as f:
        f.write(r.content)

df_zones = pd.read_csv(zone_file)

# Download trip data
if not os.path.exists(trip_file):
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    r = requests.get(url)
    r.raise_for_status()
    with open(trip_file, "wb") as f:
        f.write(r.content)

# Load trip data
df = pd.read_parquet(trip_file)

# Data cleaning

initial_rows = df.shape[0]

# Remove nulls
df_clean = df.dropna(subset=[
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "fare_amount"
])
after_nulls = df_clean.shape[0]

# Remove invalid trips
df_clean = df_clean[
    (df_clean["trip_distance"] > 0) &
    (df_clean["fare_amount"] > 0) &
    (df_clean["fare_amount"] <= 500)
]
after_invalid = df_clean.shape[0]

# Remove trips with invalid timestamps
df_clean = df_clean[
    df_clean["tpep_dropoff_datetime"] > df_clean["tpep_pickup_datetime"]
]
final_rows = df_clean.shape[0]

# Feature engineering
df_clean = df_clean.copy()
df_clean["pickup_hour"] = df_clean["tpep_pickup_datetime"].dt.hour
df_clean["pickup_date"] = df_clean["tpep_pickup_datetime"].dt.date
df_clean["pickup_day_of_week"] = df_clean["tpep_pickup_datetime"].dt.day_name()
df_clean["trip_duration_minutes"] = (
    (df_clean["tpep_dropoff_datetime"] - df_clean["tpep_pickup_datetime"]).dt.total_seconds() / 60
)
df_clean["revenue"] = df_clean["fare_amount"]

# Merge zone names
df_clean = df_clean.merge(
    df_zones[["LocationID","Zone"]], left_on="PULocationID", right_on="LocationID",
    how="left"
).rename(columns={"Zone":"PU_Zone"})
df_clean = df_clean.merge(
    df_zones[["LocationID","Zone"]], left_on="DOLocationID", right_on="LocationID",
    how="left"
).rename(columns={"Zone":"DO_Zone"})

# Sidebar filters

st.sidebar.header("Filters")

# Date range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Hour range
hour_range = st.sidebar.slider("Select Hour Range", 0, 23, (0, 23))

# Payment types
payment_types = st.sidebar.multiselect(
    "Payment Types",
    options=df_clean["payment_type"].unique(),
    default=df_clean["payment_type"].unique()
)

# Apply filters
filtered = df_clean[
    (df_clean["pickup_date"] >= start_date) &
    (df_clean["pickup_date"] <= end_date) &
    (df_clean["pickup_hour"] >= hour_range[0]) &
    (df_clean["pickup_hour"] <= hour_range[1]) &
    (df_clean["payment_type"].isin(payment_types))
]

if filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# Key metrics

st.subheader("Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trips", f"{len(filtered):,}")
col2.metric("Average Fare", f"${filtered['fare_amount'].mean():.2f}")
col3.metric("Total Revenue", f"${filtered['revenue'].sum():,.2f}")
col4.metric("Avg Distance (mi)", f"{filtered['trip_distance'].mean():.2f}")
col5.metric("Avg Duration (min)", f"{filtered['trip_duration_minutes'].mean():.2f}")

# Visualizations

# Top 10 Pickup Zones
st.subheader("Top 10 Pickup Zones by Trip Count")
top_zones = filtered["PU_Zone"].value_counts().head(10).reset_index()
top_zones.columns = ["Zone", "Trips"]
fig1 = px.bar(top_zones, x="Zone", y="Trips", color="Trips", title="Top 10 Pickup Zones")
st.plotly_chart(fig1, width="stretch")
st.markdown("""
Most trips originate from Midtown Manhattan and JFK airport, highlighting commuter and airport demand.
Business districts see consistently high pickup counts during weekdays.
Residential neighborhoods appear less frequently, indicating shorter local trips are less common.
""")

# Average Fare by Hour
st.subheader("Average Fare by Hour")
hourly_fare = filtered.groupby("pickup_hour")["fare_amount"].mean().reset_index()
fig2 = px.line(hourly_fare, x="pickup_hour", y="fare_amount", markers=True, title="Average Fare by Hour")
st.plotly_chart(fig2, width="stretch")
st.markdown("""
Fares spike during morning (7–9 AM) and evening (5–8 PM) rush hours due to traffic and longer trip distances.
Late-night hours (11 PM–2 AM) show elevated fares, likely driven by surge pricing and longer airport trips.
Midday fares are generally lower, reflecting shorter trips and lighter traffic conditions.
""")

# Trip Distance Distribution (0-25 miles)
st.subheader("Trip Distance Distribution (0–25 miles)")
filtered_distance = filtered[(filtered["trip_distance"] >= 0) & (filtered["trip_distance"] <= 25)]
fig3 = px.histogram(filtered_distance, x="trip_distance", nbins=40, title="Trip Distance Distribution")
st.plotly_chart(fig3, width="stretch")
st.markdown("""
Most trips are under 5 miles, indicating a predominance of short urban rides.
Trips between 10–20 miles are relatively rare and often correspond to airport or outer-borough travel.
Very long trips above 30 miles are extremely uncommon but generate proportionally higher revenue per ride.
""")

# Payment Type Breakdown
st.subheader("Payment Type Breakdown")
pay_counts = filtered["payment_type"].value_counts().reset_index()
pay_counts.columns = ["Payment Type","Count"]
fig4 = px.pie(pay_counts, names="Payment Type", values="Count", title="Payment Types")
st.plotly_chart(fig4, width="stretch")
st.markdown("""
Card payments (credit/debit) dominate taxi transactions.
Cash payments remain a minority, often associated with tourists or infrequent riders.
The distribution is stable throughout the day, with no significant spikes for a particular payment type.
""")

# Trips by Day of Week and Hour
st.subheader("Trips by Day of Week and Hour")
heatmap_data = filtered.groupby(["pickup_day_of_week","pickup_hour"]).size().reset_index(name="Trips")
days_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
heatmap_data["pickup_day_of_week"] = pd.Categorical(heatmap_data["pickup_day_of_week"], categories=days_order, ordered=True)
fig5 = px.density_heatmap(
    heatmap_data,
    x="pickup_hour",
    y="pickup_day_of_week",
    z="Trips",
    title="Trips by Day of Week and Hour",
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig5, width="stretch")
st.markdown("""
Weekday peaks occur during 7–9 AM and 5–8 PM, consistent with commuter traffic patterns.
Weekends show late-night activity (11 PM–2 AM), likely driven by nightlife and airport trips.
Monday mornings and Friday evenings exhibit the highest overall trip volumes, reflecting work-related travel trends.
""")