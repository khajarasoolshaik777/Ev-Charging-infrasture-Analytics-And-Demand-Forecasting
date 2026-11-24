"""
=============================================================================
  data_cleaning.py
  EV Charging Infrastructure Analysis — Step 1: Data Loading & Cleaning
=============================================================================
  Run:  python scripts/data_cleaning.py
=============================================================================
"""

import warnings
import numpy  as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent   # project root
DATA = BASE / "data"
OUT  = BASE / "dashboard" / "images" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load_datasets():
    print("\n▶  Loading datasets...")
    cs      = pd.read_csv(DATA / "charging_station.csv")
    cs_ml   = pd.read_csv(DATA / "charging_station_ml.csv")
    country = pd.read_csv(DATA / "country_summary.csv")
    world   = pd.read_csv(DATA / "world_summary.csv")
    ev      = pd.read_csv(DATA / "ev_models.csv")

    for name, df in [("charging_station", cs), ("charging_station_ml", cs_ml),
                     ("country_summary", country), ("world_summary", world),
                     ("ev_models", ev)]:
        print(f"   {name:<25} {df.shape[0]:>7,} rows  ×  {df.shape[1]} cols")

    return cs, cs_ml, country, world, ev


# ─────────────────────────────────────────────────────────────────────────────
#  CLEAN — charging_station
# ─────────────────────────────────────────────────────────────────────────────
def clean_charging_station(cs: pd.DataFrame) -> pd.DataFrame:
    print("\n▶  Cleaning charging_station...")

    # Fill missing power with 0 (unknown chargers)
    cs["power_kw"] = cs["power_kw"].fillna(0)
    cs["power_class"] = cs["power_class"].replace("UNKNOWN", "Unspecified")

    # Standardise text
    cs["city"]           = cs["city"].str.strip().replace("Unknown City", "Unknown")
    cs["state_province"] = cs["state_province"].str.strip().replace("UNKNOWN", "Unknown")

    # Drop exact duplicates on station id
    before = len(cs)
    cs.drop_duplicates(subset=["id"], inplace=True)
    print(f"   Duplicates removed : {before - len(cs):,}")

    # Cap outliers at 99th percentile (keep original too)
    p99 = cs["power_kw"].quantile(0.99)
    cs["power_kw_capped"] = cs["power_kw"].clip(upper=p99)
    print(f"   power_kw 99th pct  : {p99:.0f} kW  |  max: {cs['power_kw'].max():.0f} kW")

    return cs


# ─────────────────────────────────────────────────────────────────────────────
#  CLEAN — world_summary
# ─────────────────────────────────────────────────────────────────────────────
def clean_world_summary(world: pd.DataFrame) -> pd.DataFrame:
    print("\n▶  Cleaning world_summary...")

    world["max_power_kw"]    = world["max_power_kw"].fillna(world["max_power_kw"].median())
    world["median_power_kw"] = world["median_power_kw"].fillna(world["median_power_kw"].median())

    # Derived columns
    world["ports_per_station"]      = (world["port_count"] / world["station_count"]).round(2)
    world["ultra_fast_share"]       = (world["dc_ultra_station_count"] / world["station_count"]).round(4)
    world["fast_station_share_pct"] = (world["fast_station_share"] * 100).round(2)

    print(f"   Rows cleaned       : {len(world):,}")
    return world


# ─────────────────────────────────────────────────────────────────────────────
#  CLEAN — ev_models
# ─────────────────────────────────────────────────────────────────────────────
def clean_ev_models(ev: pd.DataFrame) -> pd.DataFrame:
    print("\n▶  Cleaning ev_models...")

    country_map = {
        "US": "United States", "United States": "United States",
        "DE": "Germany",       "Germany": "Germany",
        "CN": "China",         "China": "China",
        "KR": "South Korea",   "South Korea": "South Korea",
        "JP": "Japan",         "Japan": "Japan",
        "FR": "France",        "France": "France",
        "IT": "Italy",         "Italy": "Italy",
        "SE": "Sweden",        "Sweden": "Sweden",
        "UK": "United Kingdom","United Kingdom": "United Kingdom",
        "CZ": "Czech Republic","Czech Republic": "Czech Republic",
        "ES": "Spain",
        "RO/CN": "Romania/China",
        "SE/CN": "Sweden/China",
        "CN/DE": "China/Germany",
        "Vietnam": "Vietnam",
    }
    ev["origin_clean"] = ev["origin_country"].map(country_map).fillna(ev["origin_country"])

    def era(yr):
        if yr <= 2015: return "Pioneer (≤2015)"
        if yr <= 2019: return "Growth (2016-19)"
        return "Mainstream (2020+)"

    ev["era"] = ev["first_year"].apply(era)
    print(f"   Models cleaned     : {len(ev):,}  |  origins: {ev['origin_clean'].nunique()}")
    return ev


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DATA CLEANING PIPELINE")
    print("=" * 60)

    cs, cs_ml, country, world, ev = load_datasets()

    cs    = clean_charging_station(cs)
    world = clean_world_summary(world)
    ev    = clean_ev_models(ev)

    # Quick fill for cs_ml
    cs_ml["max_power_kw"]    = cs_ml["max_power_kw"].fillna(0)
    cs_ml["median_power_kw"] = cs_ml["median_power_kw"].fillna(0)
    cs_ml["fast_port_share"] = cs_ml["fast_port_share"].fillna(0)

    print("\n✔  All datasets cleaned successfully.")
    print(f"   charging_station : {len(cs):,} rows")
    print(f"   world_summary    : {len(world):,} rows")
    print(f"   ev_models        : {len(ev):,} rows")

    # Return cleaned dataframes for use by other scripts
    return cs, cs_ml, country, world, ev


if __name__ == "__main__":
    main()
