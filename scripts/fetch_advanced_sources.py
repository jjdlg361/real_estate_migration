#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import eurostat
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
META_DIR = ROOT / "data" / "metadata"

TARGET_GEOS = [
    "AT",
    "BE",
    "BG",
    "CY",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "HR",
    "HU",
    "IE",
    "IS",
    "IT",
    "LT",
    "LU",
    "LV",
    "MT",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "SE",
    "SI",
    "SK",
    "UK",
]

EUROSTAT_TABLES = [
    {
        "name": "migr_imm5prv_od",
        "code": "MIGR_IMM5PRV",
        "filter_pars": {"age": "TOTAL", "agedef": "REACH", "unit": "NR", "sex": "T"},
        "notes": "Immigration by previous usual residence (origin-destination country panel).",
    },
    {
        "name": "migr_asyappctza_od",
        "code": "MIGR_ASYAPPCTZA",
        "filter_pars": {"applicant": "FRST", "age": "TOTAL", "sex": "T", "unit": "PER"},
        "notes": "First-time asylum applicants by citizenship and destination (annual aggregated data).",
    },
    {
        "name": "avia_paoac_m_passengers",
        "code": "AVIA_PAOAC",
        "filter_pars": {
            "freq": "M",
            "unit": "PAS",
            "tra_meas": "PAS_CRD",
            "partner": TARGET_GEOS,
        },
        "notes": "Monthly air passengers by reporting airport and partner reporting country.",
    },
    {
        "name": "avia_paoac_q_passengers",
        "code": "AVIA_PAOAC",
        "filter_pars": {
            "freq": "Q",
            "unit": "PAS",
            "tra_meas": "PAS_CRD",
            "partner": TARGET_GEOS,
        },
        "notes": "Quarterly air passengers by reporting airport and partner reporting country.",
    },
    {
        "name": "tgs00077_nuts2_air_passengers",
        "code": "TGS00077",
        "filter_pars": {"tra_meas": "PAS_CRD", "unit": "THS_PAS"},
        "notes": "NUTS2 air transport passengers (thousand passengers).",
    },
    {
        "name": "tgs00099_nuts2_net_migration_rate",
        "code": "TGS00099",
        "filter_pars": {"indic_de": "CNMIGRATRT"},
        "notes": "NUTS2 crude net migration plus adjustment rate.",
    },
    {
        "name": "tgs00003_nuts2_gdp",
        "code": "TGS00003",
        "filter_pars": {"unit": "MIO_EUR"},
        "notes": "NUTS2 GDP (million EUR).",
    },
    {
        "name": "tgs00010_nuts2_unemployment",
        "code": "TGS00010",
        "filter_pars": {"isced11": "TOTAL", "sex": "T", "age": "Y15-74", "unit": "PC"},
        "notes": "NUTS2 unemployment rate (%).",
    },
    {
        "name": "tgs00096_nuts2_population",
        "code": "TGS00096",
        "filter_pars": {"unit": "NR", "age": "TOTAL", "sex": "T"},
        "notes": "NUTS2 population on 1 January (persons).",
    },
    {
        "name": "tgs00026_nuts2_disposable_income",
        "code": "TGS00026",
        "filter_pars": {"unit": "MIO_PPS_EU27_2020", "direct": "BAL", "na_item": "B6N"},
        "notes": "NUTS2 disposable income of private households (million PPS).",
    },
]

WB_INDICATORS = {
    "NY.GDP.PCAP.KD.ZG": "wb_gdp_pc_growth",
    "VC.BTL.DETH": "wb_battle_deaths",
    "SP.POP.TOTL": "wb_population",
    "SL.UEM.TOTL.ZS": "wb_unemployment",
}


def fetch_world_bank_indicator(indicator: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        url = (
            f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
            f"?format=json&per_page=20000&page={page}"
        )
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        payload = r.json()
        meta, data = payload
        for item in data:
            rows.append(
                {
                    "indicator": indicator,
                    "country_code_wb2": item.get("country", {}).get("id"),
                    "country_name": item.get("country", {}).get("value"),
                    "country_iso3": item.get("countryiso3code"),
                    "year": pd.to_numeric(item.get("date"), errors="coerce"),
                    "value": pd.to_numeric(item.get("value"), errors="coerce"),
                }
            )
        if page >= int(meta["pages"]):
            break
        page += 1
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    fetched_at = datetime.now(timezone.utc).isoformat()
    fetch_meta: dict[str, Any] = {
        "generated_at_utc": fetched_at,
        "eurostat_tables": [],
        "world_bank_files": [],
        "oecd_files": [],
    }

    for spec in EUROSTAT_TABLES:
        name = spec["name"]
        code = spec["code"]
        print(f"[eurostat] {name} ({code})")
        df = eurostat.get_data_df(code, filter_pars=spec["filter_pars"])
        out = RAW_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        fetch_meta["eurostat_tables"].append(
            {
                "name": name,
                "code": code,
                "filter_pars": spec["filter_pars"],
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "path": str(out.relative_to(ROOT)),
                "notes": spec["notes"],
            }
        )
        print(f"  -> {out.name} ({df.shape[0]} x {df.shape[1]})")

    print("[worldbank] fetching indicators")
    wb_frames = []
    for indicator, alias in WB_INDICATORS.items():
        df = fetch_world_bank_indicator(indicator)
        df["indicator_alias"] = alias
        wb_frames.append(df)
        print(f"  {indicator} -> {len(df)} rows")
    wb = pd.concat(wb_frames, ignore_index=True)
    wb_out = RAW_DIR / "worldbank_push_shocks_long.csv"
    wb.to_csv(wb_out, index=False)
    fetch_meta["world_bank_files"].append(
        {"path": str(wb_out.relative_to(ROOT)), "rows": int(len(wb)), "indicators": WB_INDICATORS}
    )

    print("[oecd] fetching RHPI (national + regional house prices)")
    oecd_url = (
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.SDD.TPS,DSD_RHPI@DF_RHPI_ALL,1.0/.?"
        "startPeriod=1990&format=csvfile"
    )
    resp = requests.get(oecd_url, timeout=240)
    resp.raise_for_status()
    oecd_df = pd.read_csv(StringIO(resp.text))
    oecd_out = RAW_DIR / "oecd_rhpi_all.csv"
    oecd_df.to_csv(oecd_out, index=False)
    fetch_meta["oecd_files"].append(
        {
            "path": str(oecd_out.relative_to(ROOT)),
            "rows": int(len(oecd_df)),
            "url": oecd_url,
            "notes": "OECD DSD_RHPI@DF_RHPI_ALL (national and regional house price indices)",
        }
    )
    print(f"  -> {oecd_out.name} ({len(oecd_df)} rows)")

    meta_path = META_DIR / "advanced_fetch_catalog.json"
    meta_path.write_text(json.dumps(fetch_meta, indent=2))
    print(f"[done] {meta_path}")


if __name__ == "__main__":
    main()
