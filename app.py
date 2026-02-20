import os
import sys
import requests
import polars as pl
import streamlit as st
import plotly.express as px
import datetime

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

if not os.path.exists(trip_file):
    with st.spinner("Downloading trip data..."):
        r = requests.get("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet")
        r.raise_for_status()
        with open(trip_file, "wb") as f:
            f.write(r.content)

if not os.path.exists(zone_file):
    with st.spinner("Downloading zone lookup..."):
        r = requests.get("https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv")
        r.raise_for_status()
        with open(zone_file, "wb") as f:
            f.write(r.content)

# Load data
@st.cache_data
def load_data(trip_file: str, zone_file: str) -> pl.DataFrame:
    cols_to_load = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID",
        "DOLocationID", "passenger_count", "trip_distance", "fare_amount",
        "tip_amount", "total_amount", "payment_type"
    ]

    try:
        df = pl.read_parquet(trip_file, columns=cols_to_load)
    except FileNotFoundError:
        st.error(f"File not found: {trip_file}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    df_zones = pl.read_csv(zone_file)

    # Data Cleaning
    critical_cols = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "PULocationID", "DOLocationID", "fare_amount"
    ]
    df = df.drop_nulls(subset=critical_cols)

    df = df.filter(
        (pl.col("fare_amount") > 0) &
        (pl.col("fare_amount") <= 500) &
        (pl.col("trip_distance") > 0) &
        (pl.col("tpep_dropoff_datetime") > pl.col("tpep_pickup_datetime"))
    )

    # Feature engineering
    df = df.with_columns([
        (
            (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
            .dt.total_seconds() / 60
        ).alias("trip_duration_minutes"),
        (
            pl.col("trip_distance") /
            ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
             .dt.total_seconds() / 3600)
        ).alias("trip_speed_mph"),
        pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
        pl.col("tpep_pickup_datetime").dt.weekday().alias("pickup_day_of_week"),
        pl.col("tpep_pickup_datetime").dt.date().alias("pickup_date"),
    ])

    # Zone joins
    df = df.join(
        df_zones.select([
            pl.col("LocationID").alias("PULocationID"),
            pl.col("Zone").alias("PU_Zone")
        ]),
        on="PULocationID", how="left"
    ).join(
        df_zones.select([
            pl.col("LocationID").alias("DOLocationID"),
            pl.col("Zone").alias("DO_Zone")
        ]),
        on="DOLocationID", how="left"
    )

    return df


df = load_data(trip_file, zone_file)

# Sidebar filters
st.sidebar.header("Filters")

min_date = datetime.date(2024, 1, 1)
max_date = datetime.date(2024, 1, 31)

start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("End Date",   max_date, min_value=min_date, max_value=max_date)

hour_range = st.sidebar.slider("Select Hour Range", 0, 23, (0, 23))

payment_options = df["payment_type"].unique().sort().to_list()
payment_types   = st.sidebar.multiselect("Payment Types", options=payment_options, default=payment_options)

# Apply filters
filtered = df.filter(
    (pl.col("pickup_date") >= start_date) &
    (pl.col("pickup_date") <= end_date) &
    (pl.col("pickup_hour") >= hour_range[0]) &
    (pl.col("pickup_hour") <= hour_range[1]) &
    (pl.col("payment_type").is_in(payment_types))
)

if filtered.is_empty():
    st.warning("No data for the selected filters.")
    st.stop()

# Key metrics — single pass
st.subheader("Key Metrics")

metrics = filtered.select(
    pl.len().alias("total_trips"),
    pl.col("fare_amount").mean().alias("avg_fare"),
    pl.col("fare_amount").sum().alias("total_revenue"),
    pl.col("trip_distance").mean().alias("avg_distance"),
    pl.col("trip_duration_minutes").mean().alias("avg_duration"),
).row(0, named=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trips",        f"{metrics['total_trips']:,}")
col2.metric("Average Fare",       f"${metrics['avg_fare']:.2f}")
col3.metric("Total Revenue",      f"${metrics['total_revenue']:,.2f}")
col4.metric("Avg Distance (mi)",  f"{metrics['avg_distance']:.2f}")
col5.metric("Avg Duration (min)", f"{metrics['avg_duration']:.2f}")

# Precompute all aggregations in parallel
lf = filtered.lazy()

top_zones_lf   = lf.group_by("PU_Zone").agg(pl.len().alias("Trips")).top_k(10, by="Trips")
hourly_fare_lf = lf.group_by("pickup_hour").agg(pl.col("fare_amount").mean().alias("avg_fare")).sort("pickup_hour")
payment_lf     = lf.group_by("payment_type").agg(pl.len().alias("Count"))
heatmap_lf     = lf.group_by("pickup_day_of_week", "pickup_hour").agg(pl.len().alias("Trips"))

top_zones, hourly_fare, payment_counts, heatmap_data = pl.collect_all(
    [top_zones_lf, hourly_fare_lf, payment_lf, heatmap_lf]
)

# Charts

# Top 10 Pickup Zones
st.subheader("Top 10 Pickup Zones by Trip Count")
fig1 = px.bar(
    top_zones.sort("Trips", descending=True),
    x="PU_Zone", y="Trips", color="Trips",
    title="Top 10 Pickup Zones by Trip Count"
)
st.plotly_chart(fig1, width=True)
st.markdown("""
Trips are highest in Midtown Center, JFK Airport, Upper East Side North and South, which is expected since
they are major commercial districts and an airport which serves many tourists.
Upper West Side is mostly residential, which explains lower trip counts.
""")

# Average Fare by Hour
st.subheader("Average Fare by Hour")
fig2 = px.line(
    hourly_fare,
    x="pickup_hour", y="avg_fare",
    markers=True,
    title="Average Fare by Hour of Day"
)
st.plotly_chart(fig2, width=True)
st.markdown("""
Fares spike in the morning from 3–5 AM which is likely due to citizens needing to reach work early in the morning.
There is also a significant increase in fares from 2–4 PM which is probably due to citizens returning home from work.
""")

# Trip Distance Distribution
st.subheader("Trip Distance Distribution (0–25 miles)")
fig3 = px.histogram(
    filtered.filter(pl.col("trip_distance").is_between(0, 25)),
    x="trip_distance", nbins=40,
    title="Trip Distance Distribution (0–25 miles)"
)
st.plotly_chart(fig3, width=True)
st.markdown("""
Majority of trips are below 5 miles, showing that most taxi rides are short trips within the city.
Trips between 10–20 miles are rare and most likely correspond to long-distance airport travel.
""")

# Payment Type Breakdown
st.subheader("Payment Type Breakdown")
fig4 = px.pie(
    payment_counts,
    names="payment_type", values="Count",
    title="Payment Type Breakdown"
)
st.plotly_chart(fig4, width=True)
st.markdown("""
Card payments dominate taxi transactions.
Cash payments are a minority and are likely associated with tourists or older demographics.
""")

# Trips by Day of Week and Hour
st.subheader("Trips by Day of Week and Hour")

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heatmap_data = heatmap_data.with_columns(
    pl.col("pickup_day_of_week")
      .replace_strict(old=list(range(1, 8)), new=weekdays, return_dtype=pl.String)
      .cast(pl.Enum(weekdays))
).sort(["pickup_day_of_week", "pickup_hour"])

fig5 = px.density_heatmap(
    heatmap_data,
    x="pickup_hour", y="pickup_day_of_week", z="Trips",
    color_continuous_scale="Viridis",
    title="Trips by Day of Week and Hour"
)
st.plotly_chart(fig5, width=True)
st.markdown("""
Pickup hours from 3–7 PM on weekdays show the highest trip counts, likely due to people returning home from work.
Early morning trips from 12–5 AM are more common on weekends, possibly due to nightlife and early airport trips.
""")