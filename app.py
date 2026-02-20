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
df = df.dropna(subset=[
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "fare_amount"
])
after_nulls = df.shape[0]

# Remove invalid trips
df = df[
    (df["trip_distance"] > 0) &
    (df["fare_amount"] > 0) &
    (df["fare_amount"] <= 500)
]
after_invalid = df.shape[0]

# Remove trips with invalid timestamps
df = df[
    df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"]
]
final_rows = df.shape[0]

# Feature engineering
df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.day_name()
df["trip_duration_minutes"] = (
    (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
)

# Merge zone names
df = df.merge(
    df_zones[["LocationID","Zone"]], left_on="PULocationID", right_on="LocationID",
    how="left"
).rename(columns={"Zone":"PU_Zone"})
df = df.merge(
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
    options=df["payment_type"].unique(),
    default=df["payment_type"].unique()
)

# Apply filters
filtered = df[
    (df["pickup_date"] >= start_date) &
    (df["pickup_date"] <= end_date) &
    (df["pickup_hour"] >= hour_range[0]) &
    (df["pickup_hour"] <= hour_range[1]) &
    (df["payment_type"].isin(payment_types))
]

if filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# Key metrics

st.subheader("Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trips", f"{len(filtered):,}")
col2.metric("Average Fare", f"${filtered['fare_amount'].mean():.2f}")
col3.metric("Total Revenue", f"${filtered['fare_amount'].sum():,.2f}")
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
Trips are highest in Midtown Center, JFK Airport, Upper East Side North and South which is expected since 
they are major commercial districts and an airport which assists a large number of tourists. Upper west side is also
mostly a residential area which is why they are shown to be on the lower end of the bar chart.
""")

# Average Fare by Hour
st.subheader("Average Fare by Hour")
hourly_fare = filtered.groupby("pickup_hour")["fare_amount"].mean().reset_index()
fig2 = px.line(hourly_fare, x="pickup_hour", y="fare_amount", markers=True, title="Average Fare by Hour")
st.plotly_chart(fig2, width="stretch")
st.markdown("""
Fares spike in the morning from 3–5 AM which is likely due to citizens needing to reach to work early in the morning.
There is also a significant increase in fares from 2–4 PM which is probably due to citizens returning home from work.
""")

# Trip Distance Distribution (0-25 miles)
st.subheader("Trip Distance Distribution (0–25 miles)")
filtered_distance = filtered[(filtered["trip_distance"] >= 0) & (filtered["trip_distance"] <= 25)]
fig3 = px.histogram(filtered_distance, x="trip_distance", nbins=40, title="Trip Distance Distribution")
st.plotly_chart(fig3, width="stretch")
st.markdown("""
Majority of trips are below 5 miles, showing that most taxi rides are short trips possibly in cities.
Trips between 10–20 miles are rare and most likely correspond to long travels from airports.
""")

# Payment Type Breakdown
st.subheader("Payment Type Breakdown")
pay_counts = filtered["payment_type"].value_counts().reset_index()
pay_counts.columns = ["Payment Type","Count"]
fig4 = px.pie(pay_counts, names="Payment Type", values="Count", title="Payment Types")
st.plotly_chart(fig4, width="stretch")
st.markdown("""
Card payments dominate the taxi transactions.
Cash payments are seen to be minority and are likely associated with tourists.
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
Pickup hours from 3-7pm on weekdays show the highest trip counts, likely due to people returning home from work.
Whereas, early morning trips from 12-5 am are more common on weekends, possibly due to people returning from nightlife activities and early morning airport trips.
""")