"""
=============================================================================
  app.py  —  EV Charging Infrastructure Dashboard
  Run:  streamlit run dashboard/app.py
=============================================================================
"""

import json
import warnings
import numpy  as np
import pandas as pd
import plotly.express     as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

import streamlit as st

# Check if running in Streamlit contexts
streamlit_context = False
try:
    streamlit_context = (getattr(st, "runtime", None) is not None and st.runtime.get_instance() is not None)
except Exception:
    streamlit_context = False

if not streamlit_context:
    try:
        print("⚠️ Please run this app with: streamlit run dashboard/app.py")
    except UnicodeEncodeError:
        print("Please run this app with: streamlit run dashboard/app.py")
    # Continue so the module can still be inspected in non-Streamlit contexts.

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

st.set_page_config(
    page_title="EV Charging Infrastructure",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .kpi-card {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    border-radius: 12px; padding: 20px 24px;
    color: white; text-align: center; margin: 6px 0;
  }
  .kpi-value { font-size: 2rem; font-weight: 700; margin: 0; }
  .kpi-label { font-size: 0.85rem; opacity: 0.85; margin: 0; }
  .urgent    { background: linear-gradient(135deg, #b71c1c, #e53935); }
  .high      { background: linear-gradient(135deg, #e65100, #ff6d00); }
  .medium    { background: linear-gradient(135deg, #827717, #f9a825); }
  h1 { color: #0d47a1; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {}

    # World enriched
    wp = DATA / "world_enriched.csv"
    if wp.exists():
        data["world"] = pd.read_csv(wp)

    # EV models clean
    ep = DATA / "ev_models_clean.csv"
    if ep.exists():
        data["ev"] = pd.read_csv(ep)

    # City priority predictions
    cp = DATA / "city_priority_predictions.csv"
    if cp.exists():
        data["cities"] = pd.read_csv(cp)

    # EDA dashboard JSON
    dp = DATA / "dashboard_data.json"
    if dp.exists():
        with open(dp) as f:
            data["eda_json"] = json.load(f)

    # ML dashboard JSON
    mp = DATA / "ml_dashboard_data.json"
    if mp.exists():
        with open(mp) as f:
            data["ml_json"] = json.load(f)

    return data


data = load_data()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ EV Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🌍 Infrastructure", "🚗 EV Models", "🤖 ML Predictions", "📊 Model Performance"],
    )
    st.markdown("---")
    st.markdown("**Data Sources**")
    st.caption("• 242K+ charging stations globally")
    st.caption("• 129 BEV model catalogue")
    st.caption("• 3 ML models compared")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: KPI card
# ─────────────────────────────────────────────────────────────────────────────
def kpi(label, value, css_class=""):
    st.markdown(f"""
    <div class="kpi-card {css_class}">
      <p class="kpi-value">{value}</p>
      <p class="kpi-label">{label}</p>
    </div>""", unsafe_allow_html=True)


def require_columns(df, required_cols, dataset_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"{dataset_name} is missing required columns: {missing}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("⚡ EV Charging Infrastructure Analysis")
    st.markdown("##### Global EV charging station analysis + city-level investment prediction")
    st.markdown("---")

    if "eda_json" in data:
        kpis = data["eda_json"].get("kpis", {})
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: kpi("Total Stations",   f"{kpis.get('total_stations',0):,}")
        with c2: kpi("Total Ports",       f"{kpis.get('total_ports',0):,}")
        with c3: kpi("Fast Stations",     f"{kpis.get('total_fast',0):,}")
        with c4: kpi("Ultra-Fast (≥150kW)", f"{kpis.get('total_ultra',0):,}", "urgent")
        with c5: kpi("Countries Covered", f"{kpis.get('total_countries',0)}")
        with c6: kpi("EV Models",         f"{kpis.get('total_ev_models',0)}")

        st.markdown("---")
        st.markdown(f"#### Global Fast Station Share: **{kpis.get('global_fast_share_pct',0):.1f}%**")
        st.progress(kpis.get('global_fast_share_pct', 0) / 100)

    else:
        st.info("Run the pipeline first: `python scripts/model_training.py`")

    st.markdown("---")
    st.markdown("""
    ### 🔑 Key Insights
    | # | Insight |
    |---|---------|
    | 1 | 🇺🇸 USA dominates with ~33% of all global charging stations |
    | 2 | ⚡ Only ~5.6% of stations are Ultra-Fast DC (≥150 kW) |
    | 3 | 🚗 SUVs represent 48% of all BEV model catalogues |
    | 4 | 🇨🇳 China leads EV model origin count |
    | 5 | 📈 BEV launches doubled in 2020–2024 vs all prior years |
    | 6 | 🌍 Large markets (US, UK, DE) sit at only 14–22% fast-charge share |
    """)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: INFRASTRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌍 Infrastructure":
    st.title("🌍 Global Charging Infrastructure")

    if "world" not in data:
        st.warning("world_enriched.csv not found. Run scripts first.")
        st.stop()

    world = data["world"]
    if not require_columns(world, ["country", "station_count", "fast_station_share"], "world"): 
        st.stop()

    # Top 15 countries
    st.subheader("Top 15 Countries by Station Count")
    top15 = world.nlargest(15, "station_count")
    fig = px.bar(top15, x="station_count", y="country", orientation="h",
                 color="fast_station_share_pct", color_continuous_scale="RdYlGn",
                 labels={"station_count": "Stations", "fast_station_share_pct": "Fast Share %"},
                 height=500)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("EV Readiness Score — Top 20")
        if "readiness_score" in world.columns:
            top20r = world.nlargest(20, "readiness_score")
            fig2 = px.bar(top20r, x="readiness_score", y="country", orientation="h",
                          color="readiness_score", color_continuous_scale="Blues",
                          height=500)
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Fast Share vs Station Count")
        plot_df = world[world["station_count"] > 50].copy()
        fig3 = px.scatter(plot_df, x="station_count", y="fast_station_share_pct",
                          size="station_count", color="median_power_kw",
                          hover_name="country", log_x=True,
                          color_continuous_scale="YlOrRd",
                          labels={"station_count": "Stations (log)", "fast_station_share_pct": "Fast %"},
                          height=500)
        st.plotly_chart(fig3, use_container_width=True)

    # Ultra-fast leaders
    st.subheader("Ultra-Fast DC Leaders (≥150 kW)")
    if "dc_ultra_station_count" in world.columns:
        ultra = world.nlargest(15, "dc_ultra_station_count")
        fig4 = px.bar(ultra, x="country", y="dc_ultra_station_count",
                      color="ultra_fast_share", color_continuous_scale="Reds",
                      labels={"dc_ultra_station_count": "Ultra-Fast Stations"})
        st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: EV MODELS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚗 EV Models":
    st.title("🚗 EV Model Catalogue Analysis")

    if "ev" not in data:
        st.warning("ev_models_clean.csv not found. Run scripts first.")
        st.stop()

    ev = data["ev"]
    if not require_columns(ev, ["body_style", "origin_clean", "first_year"], "ev"): 
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Models by Body Style")
        body = ev["body_style"].value_counts().head(8).reset_index()
        body.columns = ["body_style", "count"]
        fig = px.bar(body, x="body_style", y="count",
                     color="count", color_continuous_scale="Blues",
                     labels={"body_style": "Body Style", "count": "# Models"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Models by Country of Origin")
        if "origin_clean" in ev.columns:
            origin = ev["origin_clean"].value_counts().head(10).reset_index()
            origin.columns = ["country", "count"]
            fig2 = px.bar(origin, x="count", y="country", orientation="h",
                          color="count", color_continuous_scale="Oranges")
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Annual BEV Model Launches")
    launch = ev.groupby("first_year").size().reset_index(name="models")
    launch["cumulative"] = launch["models"].cumsum()
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Bar(x=launch["first_year"], y=launch["models"],
                          name="New Models", marker_color="#2196F3"), secondary_y=False)
    fig3.add_trace(go.Scatter(x=launch["first_year"], y=launch["cumulative"],
                              name="Cumulative", line=dict(color="#FF5722", width=2.5),
                              mode="lines+markers"), secondary_y=True)
    fig3.update_xaxes(title_text="Year")
    fig3.update_yaxes(title_text="New Models/Year", secondary_y=False)
    fig3.update_yaxes(title_text="Cumulative Models", secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)

    if "era" in ev.columns:
        st.subheader("Models by Era")
        era_counts = ev["era"].value_counts().reset_index()
        era_counts.columns = ["era", "count"]
        fig4 = px.pie(era_counts, names="era", values="count",
                      color_discrete_sequence=["#90CAF9","#FF7043","#4CAF50"])
        st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: ML PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 ML Predictions":
    st.title("🤖 City Investment Priority Predictions")

    if "cities" not in data:
        st.warning("city_priority_predictions.csv not found. Run scripts first.")
        st.stop()

    cities = data["cities"]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Cities Analysed",    f"{len(cities):,}")
    with c2: kpi("Urgent (≥70)",       f"{int((cities['composite_score']>=70).sum()):,}", "urgent")
    with c3: kpi("High (50-69)",        f"{int(((cities['composite_score']>=50)&(cities['composite_score']<70)).sum()):,}", "high")
    with c4: kpi("High Priority %",    f"{cities['ml_priority_label'].mean()*100:.1f}%")

    st.markdown("---")

    # Top 30 cities
    st.subheader("Top 30 Cities for New Station Investment")
    top30 = cities.nlargest(30, "composite_score")
    fig = px.bar(top30, x="composite_score",
                 y=top30.apply(lambda r: f"{r['city']}, {r['country_code']}", axis=1),
                 orientation="h",
                 color="composite_score", color_continuous_scale="RdYlGn_r",
                 labels={"composite_score": "Priority Score", "y": "City"},
                 height=700)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    # Map
    st.subheader("Global Priority Map")
    if "latitude" in cities.columns:
        map_df = cities.dropna(subset=["latitude","longitude"])
        fig_map = px.scatter_mapbox(
            map_df.nlargest(2000, "composite_score"),
            lat="latitude", lon="longitude",
            color="composite_score",
            size="composite_score",
            color_continuous_scale="RdYlGn_r",
            hover_name="city",
            hover_data={"composite_score": True, "fast_station_share": True, "station_count": True},
            zoom=1, height=500,
            mapbox_style="carto-positron",
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # Data table
    st.subheader("Priority Data Table")
    score_filter = st.slider("Minimum Composite Score", 0, 100, 50)
    filtered = cities[cities["composite_score"] >= score_filter].sort_values(
        "composite_score", ascending=False
    ).reset_index(drop=True)
    st.dataframe(filtered[[
        "country_code","city","composite_score","station_count",
        "fast_station_share","median_power_kw","ml_priority_proba","recommendation"
    ]].head(200), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.title("📊 ML Model Performance")

    if "ml_json" in data:
        ml = data["ml_json"]
        perf = ml.get("model_performance", {})

        c1, c2, c3 = st.columns(3)
        with c1: kpi("Logistic Regression AUC", f"{perf.get('lr_auc', 0):.4f}")
        with c2: kpi("Random Forest AUC",        f"{perf.get('rf_auc', 0):.4f}", "high")
        with c3: kpi("Gradient Boosting AUC",    f"{perf.get('gb_auc', 0):.4f}", "urgent")

        st.markdown(f"**Best Model:** {ml.get('model_used','N/A')}  |  "
                    f"**Threshold:** {ml.get('best_threshold',0):.2f}")
        st.markdown("---")

        # Feature importance
        if "feature_importance" in ml:
            st.subheader("Feature Importance (Top 15)")
            fi = pd.DataFrame(list(ml["feature_importance"].items()),
                              columns=["feature","importance"]).sort_values("importance")
            fig = px.bar(fi, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Blues")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Recommendation counts
        if "recommendation_counts" in ml:
            st.subheader("Recommendation Breakdown")
            rc = pd.DataFrame(list(ml["recommendation_counts"].items()),
                              columns=["recommendation","count"])
            fig2 = px.pie(rc, names="recommendation", values="count",
                          color_discrete_sequence=["#FF1744","#FF6D00","#FFC107","#4CAF50"])
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("Run `python scripts/model_training.py` to generate ML results.")

    # Saved chart images
    charts_dir = BASE / "dashboard" / "images" / "charts"
    ml_charts  = sorted(charts_dir.glob("ML_*.png"))
    if ml_charts:
        st.markdown("---")
        st.subheader("Saved Evaluation Charts")
        cols = st.columns(2)
        for i, chart in enumerate(ml_charts[:6]):
            with cols[i % 2]:
                st.image(str(chart), caption=chart.stem, use_column_width=True)
#.\.venv\Scripts\streamlit run dashboard/app.py
