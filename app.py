import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="NYC Taxi Visualization Dashboard",
    layout="wide"
)

st.title("🚕 NYC Taxi Trip Analysis Dashboard")

st.markdown("""
This dashboard explores NYC yellow taxi trip data, highlighting travel patterns,
fare behavior, payment trends, and temporal demand variations.  
Use the filters in the sidebar to dynamically explore the dataset.
""")

@st.cache_data
def load_data():
    df = pd.read_parquet("taxi_data.parquet")

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.day_name()

    df["trip_duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    return df

df = load_data()

st.sidebar.header("Filters")

# Date range
date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["pickup_date"].min(), df["pickup_date"].max()]
)

# Hour range
hour_range = st.sidebar.slider(
    "Select Hour Range",
    0, 23, (0, 23)
)

# Payment type
payment_types = st.sidebar.multiselect(
    "Select Payment Types",
    options=df["payment_type"].unique(),
    default=df["payment_type"].unique()
)

# Apply filters
filtered_df = df[
    (df["pickup_date"] >= date_range[0]) &
    (df["pickup_date"] <= date_range[1]) &
    (df["pickup_hour"] >= hour_range[0]) &
    (df["pickup_hour"] <= hour_range[1]) &
    (df["payment_type"].isin(payment_types))
]

st.subheader("Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Trips", f"{len(filtered_df):,}")
col2.metric("Average Fare", f"${filtered_df['fare_amount'].mean():.2f}")
col3.metric("Total Revenue", f"${filtered_df['fare_amount'].sum():,.2f}")
col4.metric("Avg Distance (mi)", f"{filtered_df['trip_distance'].mean():.2f}")
col5.metric("Avg Duration (min)", f"{filtered_df['trip_duration'].mean():.2f}")

st.markdown("---")

# r) BAR: Top 10 Pickup Zones
st.subheader("1️⃣ Top 10 Pickup Zones by Trip Count")

top_zones = (
    filtered_df["PULocationID"]
    .value_counts()
    .head(10)
    .reset_index()
)
top_zones.columns = ["Zone", "Trips"]

fig1 = px.bar(top_zones, x="Zone", y="Trips")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("""
The most active pickup zones dominate overall taxi demand, suggesting strong concentration
of travel activity in commercial and transit-heavy areas.  
These zones likely represent airports, business districts, or major residential hubs.
""")

# -------------------------------------------------

# s) LINE: Avg Fare by Hour
st.subheader("2️⃣ Average Fare by Hour of Day")

hourly_fare = (
    filtered_df.groupby("pickup_hour")["fare_amount"]
    .mean()
    .reset_index()
)

fig2 = px.line(hourly_fare, x="pickup_hour", y="fare_amount")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
Average fares tend to increase during peak commuting hours, reflecting longer
distances and higher traffic congestion.  
Late-night hours may also show elevated fares due to surcharges and reduced taxi availability.
""")

# -------------------------------------------------

# t) HISTOGRAM: Trip Distance
st.subheader("3️⃣ Distribution of Trip Distances")

fig3 = px.histogram(
    filtered_df,
    x="trip_distance",
    nbins=40
)

st.plotly_chart(fig3, use_container_width=True)

st.markdown("""
Most trips are short-distance rides under 5 miles, indicating taxis are frequently used
for local transportation.  
Long-distance trips are comparatively rare but contribute significantly to total revenue.
""")

# -------------------------------------------------

# u) PAYMENT TYPE BREAKDOWN
st.subheader("4️⃣ Payment Type Breakdown")

payment_counts = (
    filtered_df["payment_type"]
    .value_counts()
    .reset_index()
)
payment_counts.columns = ["Payment Type", "Count"]

fig4 = px.pie(payment_counts, names="Payment Type", values="Count")
st.plotly_chart(fig4, use_container_width=True)

st.markdown("""
Digital and card payments dominate taxi transactions, reflecting the widespread
adoption of cashless systems.  
Cash payments make up a smaller portion, indicating changing consumer preferences.
""")

# -------------------------------------------------

# v) HEATMAP: Trips by Day and Hour
st.subheader("5️⃣ Trips by Day of Week and Hour")

heatmap_data = (
    filtered_df
    .groupby(["day_of_week", "pickup_hour"])
    .size()
    .reset_index(name="Trips")
)

fig5 = px.density_heatmap(
    heatmap_data,
    x="pickup_hour",
    y="day_of_week",
    z="Trips"
)

st.plotly_chart(fig5, use_container_width=True)

st.markdown("""
Trip activity peaks during weekday commuting hours and weekend evenings,
revealing both work-related and leisure travel patterns.  
Weekends show stronger late-night demand compared to weekdays.
""")
