"""
=============================================================================
  feature_engineering.py
  EV Charging Infrastructure Analysis — Step 2: Feature Engineering
=============================================================================
  Run:  python scripts/feature_engineering.py
=============================================================================
"""

import warnings
import numpy  as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT  = BASE / "dashboard" / "images" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING — charging_station
# ─────────────────────────────────────────────────────────────────────────────
def engineer_charging_station(cs: pd.DataFrame) -> pd.DataFrame:
    print("\n▶  Engineering charging_station features...")

    def power_tier(kw):
        if kw <= 0:    return "Unknown"
        if kw < 7.5:   return "Slow AC"
        if kw < 22:    return "Standard AC"
        if kw < 50:    return "High AC"
        if kw < 150:   return "Fast DC"
        return               "Ultra-Fast DC"

    cs["power_tier"] = cs["power_kw"].apply(power_tier)
    print(f"   power_tier distribution:")
    print(cs["power_tier"].value_counts().to_string())
    return cs


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING — world_summary (Readiness Score)
# ─────────────────────────────────────────────────────────────────────────────
def engineer_world_summary(world: pd.DataFrame) -> pd.DataFrame:
    print("\n▶  Engineering world_summary features...")

    max_stations = world["station_count"].max()

    # EV Readiness Score (0–100)
    world["readiness_score"] = (
        0.40 * (np.log1p(world["station_count"]) / np.log1p(max_stations)) +
        0.30 * world["fast_station_share"] +
        0.30 * (world["median_power_kw"] / 150)
    ).round(4) * 100

    world["market_size"] = pd.cut(
        world["station_count"],
        bins=[0, 100, 1000, 10000, np.inf],
        labels=["Micro (<100)", "Small (100-999)", "Mid (1k-10k)", "Large (>10k)"]
    )

    print("   Top 10 countries by Readiness Score:")
    cols = ["country", "station_count", "fast_station_share", "median_power_kw", "readiness_score"]
    print(world.nlargest(10, "readiness_score")[cols].to_string(index=False))
    return world


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING — ev_models
# ─────────────────────────────────────────────────────────────────────────────
def engineer_ev_models(ev: pd.DataFrame) -> pd.DataFrame:
    print("\n▶  Engineering ev_models features...")

    ev["is_us_market"] = ev["market_regions"].str.contains("US", case=False).astype(int)
    ev["is_eu_market"] = ev["market_regions"].str.contains("EU", case=False).astype(int)
    ev["is_cn_market"] = ev["market_regions"].str.contains("CN", case=False).astype(int)
    ev["is_global"]    = (ev["is_us_market"] & ev["is_eu_market"] & ev["is_cn_market"]).astype(int)

    print(f"   US market models  : {ev['is_us_market'].sum()}")
    print(f"   EU market models  : {ev['is_eu_market'].sum()}")
    print(f"   CN market models  : {ev['is_cn_market'].sum()}")
    print(f"   Global models     : {ev['is_global'].sum()}")
    return ev


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING — city ML dataset
# ─────────────────────────────────────────────────────────────────────────────
def engineer_city_ml(cs_ml: pd.DataFrame, world: pd.DataFrame) -> pd.DataFrame:
    print("\n▶  Engineering city ML features...")

    # Merge country context
    world_ctx = world[[
        "country_code", "fast_station_share", "median_power_kw",
        "max_power_kw", "dc_ultra_station_count", "station_count"
    ]].rename(columns={
        "fast_station_share":     "country_fast_share",
        "median_power_kw":        "country_median_kw",
        "max_power_kw":           "country_max_kw",
        "dc_ultra_station_count": "country_ultra_count",
        "station_count":          "country_total_stations",
    })
    world_ctx["country_median_kw"] = world_ctx["country_median_kw"].fillna(0)
    world_ctx["country_max_kw"]    = world_ctx["country_max_kw"].fillna(0)

    df = cs_ml.merge(world_ctx, on="country_code", how="left")

    # Demand & gap features
    df["ports_per_station"] = (df["port_count"] / df["station_count"]).clip(0, 20)
    df["fast_gap"]          = (df["station_count"] - df["fast_station_count"]).clip(lower=0)
    df["fast_gap_ratio"]    = 1 - df["fast_station_share"]
    df["no_ultra_fast"]     = (df["dc_ultra_station_count"] == 0).astype(int)
    df["city_infra_share"]  = (
        df["station_count"] / df["country_total_stations"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
    df["power_deficit"]     = (df["country_median_kw"] - df["median_power_kw"]).clip(lower=0)
    df["relative_power"]    = (
        df["median_power_kw"] / df["country_median_kw"].replace(0, np.nan)
    ).fillna(1).clip(0, 5)
    df["log_station_count"] = np.log1p(df["station_count"])
    df["log_port_count"]    = np.log1p(df["port_count"])

    # Latitude band dummies
    df["lat_band"] = pd.cut(df["latitude"],
        bins=[-90, 0, 23.5, 40, 55, 66.5, 90],
        labels=["Southern", "Tropical", "Subtropical", "Temperate", "Cold", "Arctic"]
    )
    lat_dummies = pd.get_dummies(df["lat_band"], prefix="latband", drop_first=True)
    df = pd.concat([df, lat_dummies], axis=1)

    # Country tier
    def country_tier(n):
        if n >= 10000: return 3
        if n >= 1000:  return 2
        if n >= 100:   return 1
        return 0
    df["country_tier"] = df["country_total_stations"].apply(country_tier)

    # Demand-Gap Score & target
    df["demand_gap_score"] = (
        np.log1p(df["station_count"]) *
        (1 - df["fast_station_share"]) *
        (1 + df["no_ultra_fast"] * 0.3) *
        (1 + df["power_deficit"] / 100)
    )
    threshold = df["demand_gap_score"].quantile(0.75)
    df["priority"] = (df["demand_gap_score"] >= threshold).astype(int)

    df["priority_score"] = (
        (df["demand_gap_score"] - df["demand_gap_score"].min()) /
        (df["demand_gap_score"].max() - df["demand_gap_score"].min()) * 100
    ).round(1)

    print(f"   Total city rows   : {len(df):,}")
    print(f"   HIGH priority     : {df['priority'].sum():,}  ({df['priority'].mean()*100:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  EXPORT ENRICHED DATASETS
# ─────────────────────────────────────────────────────────────────────────────
def export_enriched(world, ev, df_city):
    print("\n▶  Exporting enriched datasets...")
    world.to_csv(DATA / "world_enriched.csv",    index=False)
    ev.to_csv(   DATA / "ev_models_clean.csv",   index=False)
    df_city.to_csv(DATA / "city_ml_features.csv", index=False)
    print("   world_enriched.csv")
    print("   ev_models_clean.csv")
    print("   city_ml_features.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # Import cleaned data from data_cleaning
    import sys
    sys.path.insert(0, str(BASE / "scripts"))
    from data_cleaning import main as clean_main

    cs, cs_ml, country, world, ev = clean_main()

    cs       = engineer_charging_station(cs)
    world    = engineer_world_summary(world)
    ev       = engineer_ev_models(ev)
    df_city  = engineer_city_ml(cs_ml, world)

    export_enriched(world, ev, df_city)

    print("\n✔  Feature engineering complete.")
    return cs, world, ev, df_city


if __name__ == "__main__":
    main()
