


---

ClusterIntelligence Pro

AI-Powered Customer Segmentation Dashboard (Streamlit + K-Means)


---

Overview

ClusterIntelligence Pro is an interactive customer segmentation dashboard built using Streamlit and K-Means clustering.

The application segments mall customers into meaningful groups based on:

Annual Income

Spending Score

Age


It provides visual analytics, business insights, and recommended strategies for each customer segment.


---

Key Features

Auto-tuning to find the optimal number of clusters (Silhouette-based)

Manual cluster selection mode

Interactive 2D and 3D visualizations

Elbow curve for cluster validation

Revenue Potential Index calculation

Radar profile comparison

Segment-level action recommendations

Downloadable segmented dataset



---

Dataset

Mall Customers Dataset (200 records)

Columns:

CustomerID

Gender

Age

Annual Income (k$)

Spending Score (1–100)


The dataset is embedded directly inside the application.


---

Technologies Used

Python

Streamlit

Scikit-learn

Plotly

Pandas

NumPy



---

Clustering Method

Features are standardized using StandardScaler

K-Means clustering is applied

Optimal K is selected using Silhouette Score (Auto mode)

Inertia values are calculated for Elbow analysis



---

Dashboard Sections

1. 2D Scatter Plot – Segment distribution


2. 3D View – Interactive clustering visualization


3. Elbow Curve – Optimal K detection


4. Business Insights – Revenue potential and strategy


5. Radar Chart – Normalized feature comparison


6. Data Table – Full segmented dataset with download option




---

Business Insight Layer

Each cluster is assigned:

Segment Name

Strategic Action Recommendation

Revenue Potential Index


This connects machine learning output to real business decisions.


---

How to Run

Install dependencies:

pip install streamlit pandas numpy scikit-learn plotly

Run the application:

streamlit run app.py

Open the local URL provided by Streamlit in your browser.


---

Model Evaluation

Silhouette Score is used to measure cluster quality:

Above 0.6 → Excellent

0.4 to 0.6 → Good

Below 0.4 → Fair



---

Project Structure

app.py
README.md


---

Conclusion

This project demonstrates how unsupervised machine learning can be integrated with interactive dashboards to generate actionable business intelligence from customer data.


---

If you want a shorter version for the GitHub description box, tell me.
