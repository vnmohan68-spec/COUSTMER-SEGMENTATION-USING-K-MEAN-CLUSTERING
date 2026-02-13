import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# Page Setup
# ------------------------------------------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
st.title("Mall Customer Segmentation using KMeans")
st.caption("Income vs Spending Score Behavioral Analysis")

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------
df = pd.read_csv("Mall_Customers.csv")

# ------------------------------------------------
# Internal Data Cleaning (Not Displayed)
# ------------------------------------------------
df = df.drop_duplicates()
df = df.dropna()

# Keep only required columns
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# ------------------------------------------------
# Train-Test Split
# ------------------------------------------------
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------
# Elbow Method
# ------------------------------------------------
st.subheader("Elbow Method")

inertia = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_train_scaled)
    inertia.append(model.inertia_)

fig_elbow = px.line(
    x=range(1, 11),
    y=inertia,
    markers=True,
    labels={"x": "Clusters (K)", "y": "Inertia"}
)

st.plotly_chart(fig_elbow, use_container_width=True)

# ------------------------------------------------
# Automatic K Selection (Silhouette)
# ------------------------------------------------
best_score = -1
best_k = 2

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_train_scaled)
    score = silhouette_score(X_train_scaled, labels)

    if score > best_score:
        best_score = score
        best_k = k

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)

train_labels = kmeans.predict(X_train_scaled)
test_labels = kmeans.predict(X_test_scaled)

train_sil = silhouette_score(X_train_scaled, train_labels)
test_sil = silhouette_score(X_test_scaled, test_labels)

# ------------------------------------------------
# Metrics
# ------------------------------------------------
st.subheader("Model Evaluation")
col1, col2, col3 = st.columns(3)
col1.metric("Optimal K", best_k)
col2.metric("Train Silhouette", round(train_sil, 3))
col3.metric("Test Silhouette", round(test_sil, 3))

# ------------------------------------------------
# Apply to Full Dataset
# ------------------------------------------------
X_scaled_full = scaler.fit_transform(X)
df["Cluster"] = kmeans.fit_predict(X_scaled_full)

# ------------------------------------------------
# Dynamic Segment Naming
# ------------------------------------------------
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(
    centers,
    columns=["Annual Income (k$)", "Spending Score (1-100)"]
)

centers_df["Score"] = centers_df.mean(axis=1)
centers_df = centers_df.sort_values("Score", ascending=False)

segment_map = {}

for rank, cluster_id in enumerate(centers_df.index):
    if rank == 0:
        segment_map[cluster_id] = "VIP Customers"
    elif rank == len(centers_df) - 1:
        segment_map[cluster_id] = "Low Value Customers"
    else:
        segment_map[cluster_id] = f"Segment {rank}"

df["Segment"] = df["Cluster"].map(segment_map)

# ------------------------------------------------
# 2D Visualization
# ------------------------------------------------
st.subheader("2D Segmentation")

fig2d = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Segment",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

st.plotly_chart(fig2d, use_container_width=True)

# ------------------------------------------------
# 3D Visualization
# ------------------------------------------------
st.subheader("3D Segmentation")

df_3d = df.copy()
df_3d["Customer Index"] = df_3d.index

fig3d = px.scatter_3d(
    df_3d,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    z="Customer Index",
    color="Segment",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

st.plotly_chart(fig3d, use_container_width=True)

# ------------------------------------------------
# Segment Distribution
# ------------------------------------------------
st.subheader("Segment Distribution")
st.dataframe(df["Segment"].value_counts())

# ------------------------------------------------
# Download Each Segment
# ------------------------------------------------
st.subheader("Download Segment Data")

for segment in df["Segment"].unique():
    segment_df = df[df["Segment"] == segment]

    csv = segment_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"Download {segment}",
        data=csv,
        file_name=f"{segment.replace(' ', '_')}.csv",
        mime="text/csv"
    )

# ------------------------------------------------
# Download Full Dataset
# ------------------------------------------------
full_csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Full Segmented Dataset",
    data=full_csv,
    file_name="Full_Segmented_Data.csv",
    mime="text/csv"
)