import streamlit as st
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
This dashboard explores NYC yellow taxi trip data for January 2024.  
You can filter by date, hour, and payment type to explore trip patterns, fares, and revenue.
""")

# -----------------------------
# Load data
# -----------------------------
raw_path = "data/raw"
trip_file = os.path.join(raw_path, "yellow_tripdata_2024-01.parquet")
zone_file = os.path.join(raw_path, "taxi_zone_lookup.csv")

df_trips = pd.read_parquet(trip_file)
df_zones = pd.read_csv(zone_file)

# Merge zone names for pickup and dropoff
df_trips = df_trips.merge(df_zones[["LocationID", "Zone"]],
                          left_on="PULocationID", right_on="LocationID",
                          how="left").rename(columns={"Zone": "PU_Zone"})
df_trips = df_trips.merge(df_zones[["LocationID", "Zone"]],
                          left_on="DOLocationID", right_on="LocationID",
                          how="left").rename(columns={"Zone": "DO_Zone"})

# -----------------------------
# Feature engineering
# -----------------------------
df_trips["pickup_hour"] = df_trips["tpep_pickup_datetime"].dt.hour
df_trips["pickup_date"] = df_trips["tpep_pickup_datetime"].dt.date
df_trips["pickup_day_of_week"] = df_trips["tpep_pickup_datetime"].dt.day_name()
df_trips["trip_duration_minutes"] = (
    df_trips["tpep_dropoff_datetime"] - df_trips["tpep_pickup_datetime"]
).dt.total_seconds() / 60
df_trips["revenue"] = df_trips["fare_amount"]

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    [df_trips["pickup_date"].min(), df_trips["pickup_date"].max()]
)
hour_range = st.sidebar.slider(
    "Select Hour Range", 0, 23, (0, 23)
)
payment_types = st.sidebar.multiselect(
    "Select Payment Types",
    options=df_trips["payment_type"].unique(),
    default=df_trips["payment_type"].unique()
)

# Apply filters
filtered = df_trips[
    (df_trips["pickup_date"] >= date_range[0]) &
    (df_trips["pickup_date"] <= date_range[1]) &
    (df_trips["pickup_hour"] >= hour_range[0]) &
    (df_trips["pickup_hour"] <= hour_range[1]) &
    (df_trips["payment_type"].isin(payment_types))
]

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
# 1️⃣ Bar chart: Top 10 Pickup Zones
# -----------------------------
st.subheader("Top 10 Pickup Zones by Trip Count")
top_zones = filtered["PU_Zone"].value_counts().head(10).reset_index()
top_zones.columns = ["Zone", "Trips"]
fig1 = px.bar(top_zones, x="Zone", y="Trips", color="Trips", title="Top 10 Pickup Zones")
st.plotly_chart(fig1, use_container_width=True)
st.markdown("Most trips originate from major transport hubs or business districts.")

# -----------------------------
# 2️⃣ Line chart: Average Fare by Hour
# -----------------------------
st.subheader("Average Fare by Hour of Day")
hourly_fare = filtered.groupby("pickup_hour")["fare_amount"].mean().reset_index()
fig2 = px.line(hourly_fare, x="pickup_hour", y="fare_amount", markers=True,
               title="Average Fare by Hour")
st.plotly_chart(fig2, use_container_width=True)
st.markdown("Fares peak during commuting hours and late night due to traffic and surcharges.")

# -----------------------------
# 3️⃣ Histogram: Trip Distance
# -----------------------------
st.subheader("Distribution of Trip Distances")
fig3 = px.histogram(filtered, x="trip_distance", nbins=40, title="Trip Distance Distribution")
st.plotly_chart(fig3, use_container_width=True)
st.markdown("Most trips are short-distance rides under 5 miles; longer trips are rare but profitable.")

# -----------------------------
# 4️⃣ Pie chart: Payment Type Breakdown
# -----------------------------
st.subheader("Payment Type Breakdown")
pay_counts = filtered["payment_type"].value_counts().reset_index()
pay_counts.columns = ["Payment Type", "Count"]
fig4 = px.pie(pay_counts, names="Payment Type", values="Count", title="Payment Types")
st.plotly_chart(fig4, use_container_width=True)
st.markdown("Card payments dominate taxi transactions; cash payments are a minority.")

# -----------------------------
# 5️⃣ Heatmap: Trips by Day of Week and Hour
# -----------------------------
st.subheader("Trips by Day of Week and Hour")
heatmap_data = filtered.groupby(["pickup_day_of_week", "pickup_hour"]).size().reset_index(name="Trips")
# Sort days
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heatmap_data["pickup_day_of_week"] = pd.Categorical(heatmap_data["pickup_day_of_week"], categories=day_order, ordered=True)
fig5 = px.density_heatmap(heatmap_data, x="pickup_hour", y="pickup_day_of_week", z="Trips",
                          title="Trips by Day and Hour")
st.plotly_chart(fig5, use_container_width=True)
st.markdown("Weekday peaks occur during commuting hours; weekends show late-night travel patterns.")
