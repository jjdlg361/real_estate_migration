#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import eurostat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
META_DIR = ROOT / "data" / "metadata"


TABLES = [
    {
        "name": "prc_hpi_a_idx",
        "code": "PRC_HPI_A",
        "filter_pars": {"purchase": "TOTAL", "unit": "I15_A_AVG"},
        "notes": "Annual house price index (2015=100), all dwellings.",
    },
    {
        "name": "prc_hpi_a_growth",
        "code": "PRC_HPI_A",
        "filter_pars": {"purchase": "TOTAL", "unit": "RCH_A_AVG"},
        "notes": "Annual house price growth rate (%), all dwellings.",
    },
    {
        "name": "prc_hpi_q_idx",
        "code": "PRC_HPI_Q",
        "filter_pars": {"purchase": "TOTAL", "unit": "I15_Q"},
        "notes": "Quarterly house price index (2015=100), all dwellings.",
    },
    {
        "name": "prc_hpi_q_growth",
        "code": "PRC_HPI_Q",
        "filter_pars": {"purchase": "TOTAL", "unit": "RCH_Q"},
        "notes": "Quarterly house price growth rate (% q/q), all dwellings.",
    },
    {
        "name": "avia_paoc_a_passengers",
        "code": "AVIA_PAOC",
        "filter_pars": {
            "freq": "A",
            "unit": "PAS",
            "tra_meas": "PAS_CRD",
            "tra_cov": "TOTAL",
            "schedule": "TOT",
        },
        "notes": "Annual air passengers carried (country-level).",
    },
    {
        "name": "avia_paoc_q_passengers",
        "code": "AVIA_PAOC",
        "filter_pars": {
            "freq": "Q",
            "unit": "PAS",
            "tra_meas": "PAS_CRD",
            "tra_cov": "TOTAL",
            "schedule": "TOT",
        },
        "notes": "Quarterly air passengers carried (country-level).",
    },
    {
        "name": "tps00019_net_migration_rate",
        "code": "TPS00019",
        "filter_pars": {"indic_de": "CNMIGRATRT"},
        "notes": "Crude rate of net migration plus adjustment (per 1,000 persons).",
    },
    {
        "name": "tps00176_immigration",
        "code": "TPS00176",
        "filter_pars": {
            "citizen": "TOTAL",
            "agedef": "REACH",
            "age": "TOTAL",
            "unit": "NR",
            "sex": "T",
        },
        "notes": "Immigration (number of persons).",
    },
    {
        "name": "tps00177_emigration",
        "code": "TPS00177",
        "filter_pars": {
            "citizen": "TOTAL",
            "agedef": "REACH",
            "age": "TOTAL",
            "unit": "NR",
            "sex": "T",
        },
        "notes": "Emigration (number of persons).",
    },
    {
        "name": "tec00115_gdp_pc_growth",
        "code": "TEC00115",
        "filter_pars": {"unit": "CLV_PCH_PRE_HAB", "na_item": "B1GQ"},
        "notes": "Real GDP per capita growth rate (%).",
    },
    {
        "name": "une_rt_a_unemployment",
        "code": "UNE_RT_A",
        "filter_pars": {"age": "Y15-74", "unit": "PC_ACT", "sex": "T"},
        "notes": "Unemployment rate (% active population), age 15-74, total.",
    },
    {
        "name": "prc_hicp_aind_inflation",
        "code": "PRC_HICP_AIND",
        "filter_pars": {"unit": "RCH_A_AVG", "coicop": "CP00"},
        "notes": "HICP annual inflation rate (%), all-items.",
    },
    {
        "name": "tps00001_population",
        "code": "TPS00001",
        "filter_pars": {"indic_de": "JAN"},
        "notes": "Population on 1 January (persons).",
    },
    {
        "name": "irt_lt_mcby_a_long_rate",
        "code": "IRT_LT_MCBY_A",
        "filter_pars": {"int_rt": "MCBY"},
        "notes": "Long-term interest rates (% p.a., Maastricht criterion bond yields).",
    },
]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()

    table_metadata = []
    for spec in TABLES:
        name = spec["name"]
        code = spec["code"]
        filter_pars = spec["filter_pars"]
        print(f"[fetch] {name} ({code}) ...")
        df = eurostat.get_data_df(code, filter_pars=filter_pars)
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(f"Unexpected response type for {code}: {type(df)!r}")
        out_csv = RAW_DIR / f"{name}.csv"
        df.to_csv(out_csv, index=False)
        meta = {
            "name": name,
            "code": code,
            "filter_pars": filter_pars,
            "notes": spec.get("notes", ""),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "fetched_at_utc": fetched_at,
            "raw_csv": str(out_csv.relative_to(ROOT)),
        }
        table_metadata.append(meta)
        print(f"  -> {out_csv.name} ({df.shape[0]} x {df.shape[1]})")

    catalog = {
        "generated_at_utc": fetched_at,
        "source": "Eurostat via eurostat Python package",
        "tables": table_metadata,
    }
    catalog_path = META_DIR / "eurostat_fetch_catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2))
    print(f"[done] Wrote metadata catalog: {catalog_path}")


if __name__ == "__main__":
    main()
