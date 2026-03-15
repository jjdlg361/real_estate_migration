#!/usr/bin/env python3
from __future__ import annotations

import json
import numbers
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from shapely.geometry import Point, shape
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
META_DIR = ROOT / "data" / "metadata"

OURAIRPORTS_URL = "https://ourairports.com/data/airports.csv"
GISCO_NUTS2_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
    "NUTS_RG_01M_2021_4326_LEVL_2.geojson"
)


def ensure_download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    r = requests.get(url, timeout=240)
    r.raise_for_status()
    path.write_bytes(r.content)


def load_airports_of_interest() -> pd.DataFrame:
    route = pd.read_parquet(RAW_DIR.parent / "processed" / "airport_partner_route_monthly.parquet")
    airports = pd.DataFrame({"rep_airp": sorted(route["rep_airp"].astype(str).unique())})
    airports["country"] = airports["rep_airp"].str[:2]
    airports["icao"] = airports["rep_airp"].str.split("_", n=1).str[1]
    return airports


def load_ourairports_catalog(path: Path) -> pd.DataFrame:
    oa = pd.read_csv(path, low_memory=False)
    cols = [c for c in ["ident", "gps_code", "iata_code", "name", "municipality", "iso_country", "latitude_deg", "longitude_deg", "type", "scheduled_service"] if c in oa.columns]
    oa = oa[cols].copy()
    for c in ["ident", "gps_code", "iata_code", "iso_country"]:
        if c in oa.columns:
            oa[c] = oa[c].astype(str)
    oa["latitude_deg"] = pd.to_numeric(oa.get("latitude_deg"), errors="coerce")
    oa["longitude_deg"] = pd.to_numeric(oa.get("longitude_deg"), errors="coerce")
    return oa


def match_airports_to_coords(airports: pd.DataFrame, oa: pd.DataFrame) -> pd.DataFrame:
    # Prefer GPS/ICAO code match, then ident.
    oa_valid = oa.dropna(subset=["latitude_deg", "longitude_deg"]).copy()
    oa_valid = oa_valid[oa_valid["type"].astype(str) != "closed"].copy() if "type" in oa_valid.columns else oa_valid

    gps = (
        oa_valid.dropna(subset=["gps_code"])
        .sort_values(["gps_code"])
        .drop_duplicates("gps_code", keep="first")
        .rename(columns={"gps_code": "icao"})
    )
    ident = (
        oa_valid.dropna(subset=["ident"])
        .sort_values(["ident"])
        .drop_duplicates("ident", keep="first")
        .rename(columns={"ident": "icao"})
    )

    m = airports.merge(
        gps[["icao", "name", "municipality", "iso_country", "latitude_deg", "longitude_deg", "type"]],
        on="icao",
        how="left",
        suffixes=("", "_gps"),
    )
    missing = m["latitude_deg"].isna()
    if missing.any():
        m2 = m.loc[missing, ["rep_airp", "country", "icao"]].merge(
            ident[["icao", "name", "municipality", "iso_country", "latitude_deg", "longitude_deg", "type"]],
            on="icao",
            how="left",
        )
        for col in ["name", "municipality", "iso_country", "latitude_deg", "longitude_deg", "type"]:
            m.loc[missing, col] = m2[col].values

    m["coord_match"] = m["latitude_deg"].notna().map({True: "matched", False: "unmatched"})
    return m


def load_nuts2_features(geojson_path: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    gj = json.loads(geojson_path.read_text())
    by_country: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for feat in gj["features"]:
        props = feat.get("properties", {})
        nuts_id = props.get("NUTS_ID")
        cntr = props.get("CNTR_CODE") or (nuts_id[:2] if isinstance(nuts_id, str) else None)
        if not nuts_id or not cntr:
            continue
        geom = shape(feat["geometry"])
        rec = {"nuts2": nuts_id, "country": cntr, "geom": geom}
        by_country.setdefault(cntr, []).append(rec)
        all_rows.append(rec)
    return by_country, all_rows


def assign_nuts2(cw: pd.DataFrame, nuts_by_country: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    trees: dict[str, tuple[STRtree, list[Any], dict[int, dict[str, Any]]]] = {}
    for cntr, recs in nuts_by_country.items():
        geoms = [r["geom"] for r in recs]
        tree = STRtree(geoms)
        geom_map = {id(g): r for g, r in zip(geoms, recs)}
        trees[cntr] = (tree, geoms, geom_map)

    for r in cw.itertuples(index=False):
        rec = {
            "rep_airp": r.rep_airp,
            "country": r.country,
            "icao": r.icao,
            "airport_name": getattr(r, "name", None),
            "municipality": getattr(r, "municipality", None),
            "latitude_deg": getattr(r, "latitude_deg", None),
            "longitude_deg": getattr(r, "longitude_deg", None),
            "coord_match": getattr(r, "coord_match", None),
            "nuts2": None,
            "crosswalk_method": None,
            "distance_deg": None,
        }
        lat = rec["latitude_deg"]
        lon = rec["longitude_deg"]
        cntr = rec["country"]
        if pd.isna(lat) or pd.isna(lon) or cntr not in trees:
            rows.append(rec)
            continue
        point = Point(float(lon), float(lat))
        tree, geoms, geom_map = trees[cntr]
        candidates = tree.query(point)
        chosen = None
        # Shapely 2 may return indices or geometries depending on version/build.
        if len(candidates):
            first = candidates[0]
            if isinstance(first, numbers.Integral):
                cand_geoms = [geoms[i] for i in candidates]
            else:
                cand_geoms = list(candidates)
            for g in cand_geoms:
                if g.contains(point) or g.touches(point):
                    chosen = geom_map[id(g)]
                    rec["crosswalk_method"] = "contains"
                    rec["distance_deg"] = 0.0
                    break
        if chosen is None:
            # Fallback: nearest polygon within country.
            dists = []
            for g in geoms:
                dists.append((g.distance(point), g))
            dists.sort(key=lambda x: x[0])
            if dists:
                dist, g = dists[0]
                chosen = geom_map[id(g)]
                rec["crosswalk_method"] = "nearest_country_polygon"
                rec["distance_deg"] = float(dist)
        if chosen is not None:
            rec["nuts2"] = chosen["nuts2"]
        rows.append(rec)

    out = pd.DataFrame(rows)
    return out


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    oa_path = RAW_DIR / "ourairports_airports.csv"
    nuts_path = RAW_DIR / "gisco_nuts2_2021_4326.geojson"
    ensure_download(OURAIRPORTS_URL, oa_path)
    ensure_download(GISCO_NUTS2_URL, nuts_path)

    airports = load_airports_of_interest()
    oa = load_ourairports_catalog(oa_path)
    matched = match_airports_to_coords(airports, oa)
    nuts_by_country, _ = load_nuts2_features(nuts_path)
    crosswalk = assign_nuts2(matched, nuts_by_country)

    out_csv = META_DIR / "airport_nuts2_crosswalk.csv"
    crosswalk.to_csv(out_csv, index=False)

    summary = {
        "airports_total": int(len(crosswalk)),
        "coords_matched": int(crosswalk["latitude_deg"].notna().sum()),
        "nuts2_matched": int(crosswalk["nuts2"].notna().sum()),
        "match_rate_nuts2": float(crosswalk["nuts2"].notna().mean()) if len(crosswalk) else None,
        "methods": crosswalk["crosswalk_method"].value_counts(dropna=False).to_dict(),
    }
    (META_DIR / "airport_nuts2_crosswalk_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[done] {out_csv}")


if __name__ == "__main__":
    main()
