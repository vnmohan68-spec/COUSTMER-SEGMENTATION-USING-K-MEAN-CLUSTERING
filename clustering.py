import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Universal Clustering App", layout="wide")
st.title("Advanced Universal Clustering System")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    # -----------------------------
    # Load Data
    # -----------------------------
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Auto Detect Numeric Columns
    # -----------------------------
    df = df.apply(pd.to_numeric, errors="ignore")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns.")
        st.stop()

    selected_features = st.multiselect(
        "Select Features for Clustering",
        numeric_cols,
        default=numeric_cols[:3]
    )

    if len(selected_features) < 2:
        st.warning("Select at least 2 features.")
        st.stop()

    # -----------------------------
    # Preprocessing
    # -----------------------------
    X = df[selected_features].copy()

    # Convert to numeric safely
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop fully empty rows
    X.dropna(how="all", inplace=True)

    # Fill remaining NaN
    X = X.fillna(X.mean())

    # Remove constant columns
    X = X.loc[:, X.var() > 0]

    if X.shape[1] < 2:
        st.error("Not enough valid features after cleaning.")
        st.stop()

    # -----------------------------
    # Outlier Detection
    # -----------------------------
    st.sidebar.header("Outlier Detection")
    contamination = st.sidebar.slider("Contamination Rate", 0.0, 0.2, 0.05)

    if contamination > 0:
        iso = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso.fit_predict(X)
        X = X[outliers == 1]
        st.success("Outliers Removed")

    # -----------------------------
    # Scaling Selection
    # -----------------------------
    st.sidebar.header("Scaling Method")
    scaling_method = st.sidebar.selectbox(
        "Choose Scaler",
        ["StandardScaler", "MinMaxScaler", "RobustScaler"]
    )

    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Algorithm Selection
    # -----------------------------
    st.sidebar.header("Clustering Algorithm")
    algorithm = st.sidebar.selectbox(
        "Choose Algorithm",
        ["KMeans", "Agglomerative", "DBSCAN"]
    )

    # -----------------------------
    # Automatic K Detection
    # -----------------------------
    if algorithm in ["KMeans", "Agglomerative"]:
        max_k = min(10, len(X_scaled)-1)
        silhouette_scores = []

        for k in range(2, max_k):
            if algorithm == "KMeans":
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                model = AgglomerativeClustering(n_clusters=k)

            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)

        optimal_k = np.argmax(silhouette_scores) + 2
        st.success(f"Optimal K detected: {optimal_k}")

        if algorithm == "KMeans":
            model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        else:
            model = AgglomerativeClustering(n_clusters=optimal_k)

        clusters = model.fit_predict(X_scaled)

    else:
        eps = st.sidebar.slider("DBSCAN eps", 0.1, 2.0, 0.5)
        min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = model.fit_predict(X_scaled)

    # -----------------------------
    # Attach Clusters
    # -----------------------------
    X["Cluster"] = clusters
    df_filtered = X.copy()

    # -----------------------------
    # Evaluation
    # -----------------------------
    if len(set(clusters)) > 1 and algorithm != "DBSCAN":
        score = silhouette_score(X_scaled, clusters)
        st.metric("Silhouette Score", round(score, 4))

    # -----------------------------
    # 2D Visualization
    # -----------------------------
    st.subheader("2D Cluster Visualization")

    fig2d = px.scatter(
        df_filtered,
        x=selected_features[0],
        y=selected_features[1],
        color="Cluster"
    )
    st.plotly_chart(fig2d, use_container_width=True)

    # -----------------------------
    # 3D Visualization (if possible)
    # -----------------------------
    if len(selected_features) >= 3:
        st.subheader("3D Cluster Visualization")

        fig3d = px.scatter_3d(
            df_filtered,
            x=selected_features[0],
            y=selected_features[1],
            z=selected_features[2],
            color="Cluster"
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # -----------------------------
    # Cluster Profiling
    # -----------------------------
    st.subheader("Cluster Profiling (Mean Values)")
    profile = df_filtered.groupby("Cluster").mean()
    st.dataframe(profile)

    # -----------------------------
    # Download Clustered Data
    # -----------------------------
    st.subheader("Download Clustered Dataset")
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        "clustered_output.csv",
        "text/csv"
    )

    # -----------------------------
    # Export Model (KMeans only)
    # -----------------------------
    if algorithm == "KMeans":
        st.subheader("Download Trained Model")

        model_data = {
            "model": model,
            "scaler": scaler,
            "features": selected_features
        }

        pickle_bytes = pickle.dumps(model_data)

        st.download_button(
            "Download Model (.pkl)",
            pickle_bytes,
            "clustering_model.pkl"
        )

else:
    st.info("Upload a CSV file to begin.")