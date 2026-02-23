# =============================================================
#  CUSTOMER SEGMENTATION - FULL CODE
#  File 1: app.py  (Streamlit Dashboard)
#  Run: streamlit run app.py
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from io import StringIO
# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="shopbags",
    layout="wide",
)
# -------------------------------------------------------------
# DATASET (Mall Customers - 200 records, hardcoded)
# -------------------------------------------------------------
CSV_DATA = """CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Female,21,16,81
3,Male,20,17,6
4,Female,23,18,77
5,Male,31,19,40
6,Female,22,20,76
7,Male,35,21,6
8,Female,23,22,94
9,Male,64,23,3
10,Female,30,24,72
11,Male,67,25,70
12,Female,35,26,13
13,Male,58,27,14
14,Female,24,28,16
15,Male,37,29,17
16,Female,22,30,35
17,Male,35,31,7
18,Female,20,32,9
19,Male,52,33,14
20,Female,35,34,40
21,Male,35,35,40
22,Female,25,36,81
23,Male,46,37,87
24,Female,31,38,97
25,Male,54,39,40
26,Female,29,40,76
27,Male,45,41,6
28,Female,35,42,94
29,Male,40,43,72
30,Female,23,44,14
31,Male,60,45,40
32,Female,21,46,81
33,Male,53,47,6
34,Female,18,48,77
35,Male,49,49,40
36,Female,21,50,76
37,Male,42,51,40
38,Female,30,52,61
39,Male,36,53,40
40,Female,20,54,44
41,Male,65,55,40
42,Female,24,56,9
43,Male,48,57,98
44,Female,31,58,40
45,Male,49,59,97
46,Female,24,60,40
47,Male,50,61,42
48,Female,27,62,40
49,Male,29,63,7
50,Female,31,64,40
51,Male,49,65,40
52,Female,24,66,18
53,Male,48,67,72
54,Female,20,68,34
55,Male,23,69,11
56,Female,25,70,40
57,Male,45,71,55
58,Female,28,72,17
59,Male,43,73,40
60,Female,22,74,76
61,Male,29,75,40
62,Female,34,76,36
63,Male,39,77,40
64,Female,32,78,61
65,Male,43,79,40
66,Female,38,80,40
67,Male,38,81,3
68,Female,47,82,40
69,Male,35,83,6
70,Female,45,84,40
71,Male,32,85,40
72,Female,26,86,20
73,Male,32,87,81
74,Female,28,88,97
75,Male,33,89,40
76,Female,18,90,73
77,Male,31,91,9
78,Female,29,92,40
79,Male,31,93,14
80,Female,29,94,40
81,Male,57,95,40
82,Female,23,96,89
83,Male,32,97,40
84,Female,21,98,79
85,Male,38,99,40
86,Female,40,101,77
87,Male,29,102,40
88,Female,25,103,56
89,Male,41,104,40
90,Female,31,105,40
91,Male,42,106,40
92,Female,24,107,40
93,Male,32,108,81
94,Female,31,109,40
95,Male,34,110,92
96,Female,28,111,40
97,Male,35,112,40
98,Female,21,113,15
99,Male,38,114,40
100,Female,25,115,40
101,Male,29,116,40
102,Female,32,117,40
103,Male,35,118,40
104,Female,36,119,40
105,Male,28,120,40
106,Female,32,121,40
107,Male,27,122,40
108,Female,39,123,40
109,Male,33,124,40
110,Female,29,125,40
111,Male,30,126,40
112,Female,35,127,40
113,Male,26,128,40
114,Female,34,129,40
115,Male,42,130,40
116,Female,23,131,40
117,Male,35,132,40
118,Female,28,133,40
119,Male,23,134,40
120,Female,36,135,40
121,Male,44,136,40
122,Female,32,137,40
123,Male,28,138,40
124,Female,29,139,40
125,Male,35,140,40
126,Female,38,141,40
127,Male,44,142,40
128,Female,30,143,40
129,Male,26,144,40
130,Female,24,145,40
131,Male,37,146,40
132,Female,28,147,40
133,Male,39,148,40
134,Female,31,149,40
135,Male,29,150,40
136,Female,45,126,28
137,Male,32,126,74
138,Female,32,137,18
139,Male,27,137,83
140,Female,30,138,20
141,Male,33,138,76
142,Female,29,139,17
143,Male,32,139,85
144,Female,30,140,27
145,Male,31,140,78
146,Female,36,141,23
147,Male,43,141,71
148,Female,23,142,35
149,Male,39,142,55
150,Female,40,143,59
151,Male,22,143,66
152,Female,42,144,22
153,Male,29,144,79
154,Female,32,145,29
155,Male,28,145,73
156,Female,46,146,24
157,Male,33,146,72
158,Female,30,147,32
159,Male,31,147,71
160,Female,26,148,35
161,Male,28,148,65
162,Female,45,149,28
163,Male,24,149,72
164,Female,22,150,35
165,Male,44,150,62
166,Female,46,151,28
167,Male,33,151,72
168,Female,36,152,33
169,Male,28,152,73
170,Female,34,153,29
171,Male,37,153,73
172,Female,31,154,36
173,Male,26,154,77
174,Female,29,155,39
175,Male,33,155,49
176,Female,20,156,47
177,Male,23,156,53
178,Female,45,157,16
179,Male,34,157,73
180,Female,28,158,41
181,Male,42,158,57
182,Female,39,159,49
183,Male,23,159,74
184,Female,25,160,42
185,Male,43,160,59
186,Female,38,161,48
187,Male,30,161,58
188,Female,22,162,21
189,Male,45,162,76
190,Female,35,163,44
191,Male,29,163,57
192,Female,21,164,46
193,Male,34,164,57
194,Female,26,165,40
195,Male,28,165,60
196,Female,35,166,40
197,Male,27,166,62
198,Female,22,167,43
199,Male,26,167,57
200,Female,41,168,58"""
# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------
COLORS = [
    "#00ffe0", "#a855f7", "#ff6b2b", "#ff2d6b", "#4da6ff",
    "#ffd166", "#7bed9f", "#ff9ff3", "#54a0ff", "#ff6348",
]
SEGMENT_NAMES = [
    "Premium Targets",       "Growth Potential",       "High-Value Loyalists",
    "Affluent Conservatives","Budget Engagers",        "Rising Stars",
    "Core Customers",        "Power Buyers",           "VIP Elite",
    "Casual Browsers",
]
SEGMENT_EMOJI = ["💎","🌱","🏆","💰","⚡","🚀","🎯","🔥","🌟","🛍️"]
ACTIONS = [
    "Upsell premium products & loyalty rewards",
    "Nurture with targeted offers & re-engagement",
    "Retain with exclusive perks & priority service",
    "Introduce luxury bundles & personalized outreach",
    "Drive volume with flash deals & promotions",
    "Invest in retention before churn risk grows",
    "Maintain satisfaction & cross-sell adjacents",
    "Lock in with subscription & membership plans",
    "White-glove service & referral programs",
    "Convert with entry-level offers & samples",
]
# -------------------------------------------------------------
# CSS  (dark neon theme)
# -------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?
family=Space+Mono:wght@400;700&family=Syne:wght@800&display=swap');
html, body, .stApp { background: #02050e !important; }
* { font-family: 'Space Mono', monospace; color: #eef2ff; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] {
    background: #080d1a !important;
    border-right: 1px solid #1e2a40;
}
[data-testid="stMetric"] {
    background: #080d1a;
    border: 1px solid #1e2a40;
    border-radius: 14px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] {
    color: #00ffe0 !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #6b7fa3 !important;
    font-size: 11px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00ffe0, #00b8a4) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 50px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    box-shadow: 0 0 20px rgba(0,255,224,0.3) !important;
}
.stButton > button:hover {
    box-shadow: 0 0 30px rgba(0,255,224,0.6) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #080d1a;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    color: #6b7fa3 !important;
    font-size: 12px !important;
}
.stTabs [aria-selected="true"] {
    color: #00ffe0 !important;
    border-bottom: 2px solid #00ffe0 !important;
    background: rgba(0,255,224,0.07) !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00ffe0, #a855f7) !important;
}
.stProgress > div > div { background: #1e2a40 !important; }
div[data-testid="stDataFrame"] {
    border: 1px solid #1e2a40;
    border-radius: 10px;
}
.card {
    background: #080d1a;
    border: 1px solid #1e2a40;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)
# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("## Controls")
    st.divider()
    mode = st.radio(
        "Clustering Mode",
        ["Auto-Tune (find best K)", "Manual K (you choose)"],
    )
    k_manual = 5
    if "Manual" in mode:
        k_manual = st.slider("Number of Clusters K", 2, 10, 5)
    st.divider()
    feat_options = ["Annual Income (k$)", "Spending Score (1-100)", "Age"]
    sel_features = st.multiselect(
        "Features to Cluster On",
        feat_options,
        default=["Annual Income (k$)", "Spending Score (1-100)"],
    )
    st.divider()
    st.button("Run Clustering", use_container_width=True)
    st.divider()
    st.markdown(
        "**How to use**\n"
        "1. Choose Auto or Manual mode\n"
        "2. Select features\n"
        "3. Click Run Clustering\n"
        "4. Explore the tabs"
    )
# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.markdown("""
<div style="padding:28px 0 8px;">
  <p style="font-size:9px;letter-spacing:4px;color:#00ffe0;margin:0 0 6px;">
    CUSTOMER SEGMENTATION PLATFORM
  </p>
  <h1 style="font-family:Syne,sans-serif;font-size:44px;font-weight:900;
             letter-spacing:-2px;margin:0 0 8px;">
    <span style="color:#00ffe0;">Cluster</span><span
style="color:#eef2ff;">Intelligence</span>
    <sup style="font-size:12px;color:#a855f7;font-weight:400;"> PRO</sup>
  </h1>
  <p style="color:#6b7fa3;font-size:11px;margin:0;">
    AI-powered customer segmentation on Mall Customers Dataset (200 records)
  </p>
</div>
<hr style="border:none;border-top:1px solid #1e2a40;margin:12px 0 20px;">
""", unsafe_allow_html=True)
# Guard
if not sel_features:
    st.warning("Please select at least one feature from the sidebar.")
    st.stop()
# -------------------------------------------------------------
# CLUSTERING  (cached — re-runs only when inputs change)
# -------------------------------------------------------------
@st.cache_data
def load_and_cluster(features_key, mode_key, k_key):
    df = pd.read_csv(StringIO(CSV_DATA))
    features = list(features_key)
    Xs = StandardScaler().fit_transform(df[features].values)
    if "Auto" in mode_key:
        best_k, best_sil = 2, -1.0
        for k in range(2, 11):
            labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
            s = silhouette_score(Xs, labels)
            if s > best_sil:
                best_sil, best_k = s, k
    else:
        best_k = k_key
        labels = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(Xs)
        best_sil = float(silhouette_score(Xs, labels))
    df["Cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(Xs)
    df["Segment"] = df["Cluster"].apply(
        lambda x: SEGMENT_EMOJI[x % len(SEGMENT_EMOJI)] + " " + SEGMENT_NAMES[x %
len(SEGMENT_NAMES)]
    )
    df["Action"] = df["Cluster"].apply(lambda x: ACTIONS[x % len(ACTIONS)])
    inertias = [
        KMeans(n_clusters=k, n_init=5, random_state=42).fit(Xs).inertia_
        for k in range(2, 11)
    ]
    return df, best_k, float(best_sil), inertias
df, best_k, best_sil, inertias = load_and_cluster(
    tuple(sel_features), mode, k_manual
)
# -------------------------------------------------------------
# KPI METRICS
# -------------------------------------------------------------
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Customers",  str(len(df)))
m2.metric("Segments",   str(best_k))
m3.metric("Avg Income", str(round(df["Annual Income (k$)"].mean(), 1)) + "k")
m4.metric("Avg Spend",  str(round(df["Spending Score (1-100)"].mean(), 1)))
m5.metric("Avg Age",    str(round(df["Age"].mean(), 1)))
m6.metric("Silhouette", str(round(best_sil, 3)))
st.markdown("<br>", unsafe_allow_html=True)
# -------------------------------------------------------------
# CHART HELPERS  (no xaxis/yaxis in base dict to avoid conflicts)
# -------------------------------------------------------------
def base_layout(title="", h=420):
    return dict(
        paper_bgcolor="#080d1a",
        plot_bgcolor="#02050e",
        font=dict(color="#8899aa", family="Space Mono", size=11),
        margin=dict(l=10, r=10, t=45, b=10),
        height=h,
        title=title,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#eef2ff", size=10)),
    )
def axis(title_text=""):
    return dict(
        title=title_text,
        gridcolor="#141c2e",
        color="#6b7fa3",
        zerolinecolor="#141c2e",
    )
# -------------------------------------------------------------
# TABS
# -------------------------------------------------------------
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📍 2D Scatter",
    "🌐 3D View",
    "📈 Elbow",
    "📊 Business",
    "🕸️ Radar",
    "📋 Data",
])
# ── Tab 1: 2D Scatter ────────────────────────────────────────
with t1:
    if len(sel_features) < 2:
        st.info("Select at least 2 features.")
    else:
        fx, fy = sel_features[0], sel_features[1]
        fig = px.scatter(
            df, x=fx, y=fy, color="Segment",
            color_discrete_sequence=COLORS[:best_k],
            hover_data=["CustomerID", "Gender", "Age", "Action"],
            title="Customer Segments: " + fx + " vs " + fy,
        )
        fig.update_traces(marker=dict(size=10, opacity=0.85,
                                      line=dict(width=0.5, color="#02050e")))
        lay = base_layout()
        lay["xaxis"] = axis(fx)
        lay["yaxis"] = axis(fy)
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Hover over any point to see the recommended business action.")
# ── Tab 2: 3D Scatter ────────────────────────────────────────
with t2:
    st.markdown("#### 3D Customer Segment Explorer")
    st.caption("Drag to rotate  |  Scroll to zoom  |  Click legend to isolate segments")
    fig3d = px.scatter_3d(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        z="Age",
        color="Segment",
        color_discrete_sequence=COLORS[:best_k],
        hover_data=["CustomerID", "Gender", "Action"],
        opacity=0.85,
        title="3D View: Income x Spend Score x Age",
    )
    fig3d.update_traces(marker=dict(size=5, line=dict(width=0.3, color="#02050e")))
    fig3d.update_layout(
        paper_bgcolor="#080d1a",
        font=dict(color="#8899aa", family="Space Mono", size=10),
        margin=dict(l=0, r=0, t=45, b=0),
        height=560,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#eef2ff", size=10)),
        scene=dict(
            bgcolor="#02050e",
            xaxis=dict(title="Annual Income (k$)", backgroundcolor="#080d1a",
                       gridcolor="#1e2a40", showbackground=True, color="#6b7fa3"),
            yaxis=dict(title="Spending Score",      backgroundcolor="#080d1a",
                       gridcolor="#1e2a40", showbackground=True, color="#6b7fa3"),
            zaxis=dict(title="Age",                 backgroundcolor="#080d1a",
                       gridcolor="#1e2a40", showbackground=True, color="#6b7fa3"),
        ),
    )
    st.plotly_chart(fig3d, use_container_width=True)
# ── Tab 3: Elbow ─────────────────────────────────────────────
with t3:
    ks = list(range(2, 11))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=inertias,
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(0,255,224,0.06)",
        line=dict(color="#00ffe0", width=2.5),
        marker=dict(size=7, color="#a855f7", line=dict(color="#00ffe0", width=1.5)),
        name="Inertia (WCSS)",
    ))
    fig.add_trace(go.Scatter(
        x=[best_k], y=[inertias[best_k - 2]],
        mode="markers+text",
        marker=dict(size=16, color="#ff2d6b", symbol="star"),
        text=["  Optimal K=" + str(best_k)],
        textposition="top right",
        textfont=dict(color="#ff2d6b", size=11),
        name="Best K=" + str(best_k),
    ))
    lay = base_layout("Elbow Curve - Optimal Number of Segments")
    lay["xaxis"] = axis("Number of Segments (K)")
    lay["xaxis"]["tickmode"] = "linear"
    lay["yaxis"] = axis("Inertia (WCSS)")
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The elbow point shows where adding more clusters stops providing significantvalue.")
# ── Tab 4: Business Insights ─────────────────────────────────
with t4:
    st.markdown("#### Business Intelligence Dashboard")
    summary = df.groupby("Cluster").agg(
        Segment=("Segment",                 "first"),
        Customers=("CustomerID",            "count"),
        Avg_Age=("Age",                     "mean"),
        Avg_Income=("Annual Income (k$)",   "mean"),
        Avg_Spend=("Spending Score (1-100)","mean"),
        Action=("Action",                   "first"),
    ).round(1).reset_index(drop=True)
    summary["Revenue_Index"] = (
        (summary["Avg_Income"] / summary["Avg_Income"].max()) * 0.5 +
        (summary["Avg_Spend"]  / summary["Avg_Spend"].max())  * 0.5
    ).round(3)
    col_l, col_r = st.columns(2)
    with col_l:
        fig_bar = go.Figure(go.Bar(
            x=summary["Revenue_Index"],
            y=summary["Segment"],
            orientation="h",
            marker=dict(
                color=summary["Revenue_Index"].tolist(),
                colorscale=[[0,"#1e2a40"],[0.5,"#a855f7"],[1,"#00ffe0"]],
                line=dict(width=0),
            ),
            text=[str(round(v,2)) for v in summary["Revenue_Index"]],
            textposition="outside",
            textfont=dict(color="#eef2ff"),
        ))
        lay_bar = base_layout("Revenue Potential Index by Segment", h=380)
        lay_bar["xaxis"] = axis("Index (0-1)")
        lay_bar["yaxis"] = dict(title="", gridcolor="#141c2e",
                                color="#eef2ff", zerolinecolor="#141c2e")
        lay_bar["showlegend"] = False
        fig_bar.update_layout(**lay_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_r:
        fig_bub = px.scatter(
            summary, x="Avg_Income", y="Avg_Spend",
            size="Customers", color="Segment",
            color_discrete_sequence=COLORS[:best_k],
            text="Segment", size_max=55,
            title="Income vs Spend Score (bubble = segment size)",
        )
        fig_bub.update_traces(
            textposition="top center",
            textfont=dict(size=8, color="#eef2ff"),
            marker=dict(opacity=0.8, line=dict(width=1, color="#02050e")),
        )
        lay_bub = base_layout("", h=380)
        lay_bub["showlegend"] = False
        lay_bub["xaxis"] = axis("Avg Annual Income (k$)")
        lay_bub["yaxis"] = axis("Avg Spending Score")
        fig_bub.update_layout(**lay_bub)
        st.plotly_chart(fig_bub, use_container_width=True)
    # Segment Action Cards
    st.markdown("#### Segment Action Plan")
    num_cols  = min(best_k, 3)
    card_cols = st.columns(num_cols)
    for i, row in summary.iterrows():
        col_idx = i % num_cols
        c       = COLORS[i % len(COLORS)]
        rev_pct = int(row["Revenue_Index"] * 100)
        seg     = str(row["Segment"])
        cust    = int(row["Customers"])
        inc     = str(int(row["Avg_Income"])) + "k"
        spd     = str(int(row["Avg_Spend"]))
        act     = str(row["Action"])
        card = (
            '<div class="card" style="border-left:3px solid ' + c + ';">'
            '<div style="font-size:13px;font-weight:700;color:' + c + ';margin-bottom:6px;">'
+ seg + '</div>'
            '<div style="font-size:10px;color:#6b7fa3;margin-bottom:10px;">'
            + str(cust) + ' customers &nbsp;&middot;&nbsp; Income: $' + inc
            + ' &nbsp;&middot;&nbsp; Spend: ' + spd + '</div>'
            '<div style="background:#1e2a40;border-radius:4px;height:4px;marginbottom:10px;">'
            '<div style="background:' + c + ';height:4px;border-radius:4px;width:'
            + str(rev_pct) + '%;"></div></div>'
            '<div style="font-size:10px;color:#eef2ff;">'
            '<strong style="color:#00ffe0;">Strategy: </strong>' + act + '</div>'
            '</div>'
        )
        with card_cols[col_idx]:
            st.markdown(card, unsafe_allow_html=True)
# ── Tab 5: Radar ─────────────────────────────────────────────
with t5:
    if len(sel_features) < 2:
        st.info("Select at least 2 features for the radar chart.")
    else:
        means = df.groupby("Cluster")[sel_features].mean()
        norm  = (means - means.min()) / (means.max() - means.min() + 1e-9)
        fig   = go.Figure()
        for i, row in norm.iterrows():
            vals = list(row.values) + [row.values[0]]
            cats = sel_features + [sel_features[0]]
            hx   = COLORS[i % len(COLORS)].lstrip("#")
            r2, g2, b2 = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                fillcolor="rgba(" + str(r2) + "," + str(g2) + "," + str(b2) + ",0.15)",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                name=SEGMENT_EMOJI[i % len(SEGMENT_EMOJI)] + " Cluster " + str(i),
            ))
        fig.update_layout(
            paper_bgcolor="#080d1a",
            font=dict(color="#8899aa", family="Space Mono", size=11),
            margin=dict(l=30, r=30, t=50, b=30),
            height=460,
            title="Normalised Feature Profile by Segment",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#eef2ff", size=10)),
            polar=dict(
                bgcolor="#02050e",
                radialaxis=dict(gridcolor="#1e2a40", color="#6b7fa3",
                                linecolor="#1e2a40", range=[0,1]),
                angularaxis=dict(gridcolor="#1e2a40", color="#6b7fa3",
                                 linecolor="#1e2a40"),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
# ── Tab 6: Data ──────────────────────────────────────────────
with t6:
    st.markdown("### Segment Summary")
    disp = df.groupby("Cluster").agg(
        Segment=("Segment",                  "first"),
        Customers=("CustomerID",             "count"),
        Avg_Age=("Age",                      "mean"),
        Avg_Income=("Annual Income (k$)",    "mean"),
        Avg_Spend=("Spending Score (1-100)", "mean"),
        Recommended_Action=("Action",        "first"),
    ).round(1).reset_index(drop=True)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.markdown("### Full Customer Dataset")
    out = df[["CustomerID","Gender","Age","Annual Income (k$)",
              "Spending Score (1-100)","Cluster","Segment","Action"]].copy()
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Segmented CSV",
        out.to_csv(index=False),
        "customers_segmented.csv",
        "text/csv",
    )
# -------------------------------------------------------------
# FOOTER — Silhouette score
# -------------------------------------------------------------
st.divider()
fa, fb = st.columns([1, 4])
with fa:
    quality = "Excellent" if best_sil > 0.6 else "Good" if best_sil > 0.4 else "Fair"
    st.markdown(
        '<p style="font-size:42px;font-weight:900;color:#00ffe0;margin:0;">'
        + str(round(best_sil, 3)) + '</p>'
        '<p style="color:#6b7fa3;font-size:11px;margin:4px 0 0;">Silhouette Score - '
        + quality + '</p>',
        unsafe_allow_html=True,
    )
with fb:
    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(float(best_sil))
    st.caption("Measures cluster separation: 0 = poor, 1 = perfect")
st.markdown(
    '<p style="text-align:center;color:#1e2a40;font-size:9px;'
    'letter-spacing:3px;margin-top:30px;">'
    'CLUSTERIQ PRO - K-MEANS SEGMENTATION - MALL CUSTOMERS DATASET</p>',
    unsafe_allow_html=True,
)
