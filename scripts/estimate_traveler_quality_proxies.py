#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import eurostat
from linearmodels.panel import PanelOLS


warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"linearmodels(\..*)?")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
META_DIR = ROOT / "data" / "metadata"
PAPER_DIR = ROOT / "paper_overleaf"
PAPER_TABLES_DIR = PAPER_DIR / "tables"
PAPER_FIGS_DIR = PAPER_DIR / "figures"

Q_COL_RE = re.compile(r"^\d{4}-Q[1-4]$")
M_COL_RE = re.compile(r"^\d{4}-\d{2}$")
TARGET_GEOS = {
    "AT","BE","BG","CY","CZ","DE","DK","EE","EL","ES","FI","FR","HR","HU","IE","IS","IT","LT","LU","LV","MT","NL","NO","PL","PT","RO","SE","SI","SK","UK"
}

RAW_HICP_AIRFARE = RAW_DIR / "prc_hicp_midx_cp0733_m_airfare.csv"
RAW_APAL_PAS = RAW_DIR / "avia_tf_apal_q_pas_airline_mix.csv"
RAW_APAL_MOVE = RAW_DIR / "avia_tf_apal_q_move_airline_mix.csv"


def winsorize_series(s: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.quantile([p_low, p_high])
    return x.clip(lo, hi)


def _wide_to_long(df: pd.DataFrame, col_pattern: re.Pattern, period_name: str, value_name: str) -> pd.DataFrame:
    geo_cols = [c for c in df.columns if "\\TIME_PERIOD" in str(c)]
    if len(geo_cols) != 1:
        raise ValueError(f"Expected one geo/time column, got {geo_cols}")
    geo_col = geo_cols[0]
    time_cols = [c for c in df.columns if col_pattern.match(str(c))]
    id_cols = [c for c in df.columns if c not in time_cols]
    out = df.melt(id_vars=id_cols, value_vars=time_cols, var_name=period_name, value_name=value_name)
    dim_name = str(geo_col).split("\\")[0]
    out = out.rename(columns={geo_col: dim_name})
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out


def fetch_or_load_hicp_airfare() -> pd.DataFrame:
    if RAW_HICP_AIRFARE.exists():
        return pd.read_csv(RAW_HICP_AIRFARE)
    df = eurostat.get_data_df(
        "prc_hicp_midx",
        filter_pars={"freq": "M", "unit": "I15", "coicop": "CP0733", "geo": sorted(TARGET_GEOS)},
    )
    RAW_HICP_AIRFARE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_HICP_AIRFARE, index=False)
    return df


def fetch_or_load_apal_pas() -> pd.DataFrame:
    if RAW_APAL_PAS.exists():
        return pd.read_csv(RAW_APAL_PAS)
    df = eurostat.get_data_df(
        "AVIA_TF_APAL",
        filter_pars={
            "freq": "Q",
            "unit": "PAS",
            "tra_meas": "PAS_CRD",
            "airline": ["TOTAL", "LIC_EU", "LIC_NEU"],
        },
    )
    RAW_APAL_PAS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_APAL_PAS, index=False)
    return df


def fetch_or_load_apal_move() -> pd.DataFrame:
    if RAW_APAL_MOVE.exists():
        return pd.read_csv(RAW_APAL_MOVE)
    df = eurostat.get_data_df(
        "AVIA_TF_APAL",
        filter_pars={
            "freq": "Q",
            "unit": "MOVE",
            "tra_meas": "CACM",
            "airline": ["TOTAL", "LIC_EU", "LIC_NEU"],
        },
    )
    RAW_APAL_MOVE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_APAL_MOVE, index=False)
    return df


def build_hicp_airfare_quarterly() -> pd.DataFrame:
    raw = fetch_or_load_hicp_airfare()
    long_df = _wide_to_long(raw, M_COL_RE, "month_str", "airfare_idx")
    long_df = long_df.rename(columns={"geo": "geo"}) if "geo" in long_df.columns else long_df.rename(columns={"geo\\TIME_PERIOD": "geo"})
    if "geo" not in long_df.columns:
        geo_like = [c for c in long_df.columns if c not in {"freq", "unit", "coicop", "month_str", "airfare_idx"}]
        long_df = long_df.rename(columns={geo_like[0]: "geo"})
    long_df = long_df[long_df["geo"].astype(str).isin(TARGET_GEOS)].copy()
    long_df["month"] = pd.PeriodIndex(long_df["month_str"], freq="M")
    long_df = long_df.sort_values(["geo", "month"]).reset_index(drop=True)
    long_df["airfare_yoy_m"] = long_df.groupby("geo")["airfare_idx"].transform(lambda s: np.log(s.where(s > 0)).diff(12) * 100.0)
    long_df["quarter"] = long_df["month"].dt.asfreq("Q")
    q = (
        long_df.groupby(["geo", "quarter"], as_index=False)
        .agg(
            airfare_idx_q=("airfare_idx", "mean"),
            airfare_yoy_m_q=("airfare_yoy_m", "mean"),
        )
        .sort_values(["geo", "quarter"])
        .reset_index(drop=True)
    )
    q["airfare_yoy_q"] = q.groupby("geo")["airfare_idx_q"].transform(lambda s: np.log(s.where(s > 0)).diff(4) * 100.0)
    q["period_str"] = q["quarter"].astype(str)
    return q


def _build_apal_country_quarter(raw: pd.DataFrame, value_name: str) -> pd.DataFrame:
    long_df = _wide_to_long(raw, Q_COL_RE, "period_str", value_name)
    rep_col = next((c for c in long_df.columns if c.startswith("rep_airp")), None)
    if rep_col is None:
        raise KeyError("Missing rep_airp column in APAL extract")
    long_df = long_df.rename(columns={rep_col: "rep_airp"}).copy()
    long_df["geo"] = long_df["rep_airp"].astype(str).str[:2]
    long_df = long_df[long_df["geo"].isin(TARGET_GEOS)].copy()
    long_df = long_df.dropna(subset=[value_name]).copy()
    long_df["period_str"] = pd.PeriodIndex(long_df["period_str"].astype(str), freq="Q").astype(str)
    q = (
        long_df.groupby(["geo", "period_str", "airline"], as_index=False)[value_name]
        .sum(min_count=1)
        .sort_values(["geo", "period_str", "airline"])
        .reset_index(drop=True)
    )
    wide = q.pivot_table(index=["geo", "period_str"], columns="airline", values=value_name, aggfunc="first").reset_index()
    wide.columns = [f"{value_name}_{c.lower()}" if c not in {"geo", "period_str"} else c for c in wide.columns]
    return wide


def build_airline_quality_quarterly() -> pd.DataFrame:
    pas_raw = fetch_or_load_apal_pas()
    mov_raw = fetch_or_load_apal_move()
    pas = _build_apal_country_quarter(pas_raw, "pas")
    mov = _build_apal_country_quarter(mov_raw, "move")
    q = pas.merge(mov, on=["geo", "period_str"], how="outer")
    # Normalize columns
    for c in [
        "pas_total", "pas_lic_eu", "pas_lic_neu",
        "move_total", "move_lic_eu", "move_lic_neu",
    ]:
        if c not in q.columns:
            q[c] = np.nan
    for c, total in [
        ("pas_lic_eu", "pas_total"),
        ("pas_lic_neu", "pas_total"),
        ("move_lic_eu", "move_total"),
        ("move_lic_neu", "move_total"),
    ]:
        if c in q.columns and total in q.columns:
            q[c] = np.where(q[total].notna(), q[c].fillna(0.0), q[c])

    q["lic_neu_share_pas"] = np.where(q["pas_total"] > 0, q["pas_lic_neu"] / q["pas_total"], np.nan)
    q["lic_eu_share_pas"] = np.where(q["pas_total"] > 0, q["pas_lic_eu"] / q["pas_total"], np.nan)
    q["lic_neu_share_move"] = np.where(q["move_total"] > 0, q["move_lic_neu"] / q["move_total"], np.nan)
    q["pax_per_move_total"] = np.where(q["move_total"] > 0, q["pas_total"] / q["move_total"], np.nan)
    q["pax_per_move_lic_neu"] = np.where(q["move_lic_neu"] > 0, q["pas_lic_neu"] / q["move_lic_neu"], np.nan)
    q["pax_per_move_lic_eu"] = np.where(q["move_lic_eu"] > 0, q["pas_lic_eu"] / q["move_lic_eu"], np.nan)
    q["lic_neu_pas_growth_yoy"] = (
        q.sort_values(["geo", "period_str"])
        .groupby("geo")["pas_lic_neu"]
        .transform(lambda s: np.log(s.where(s > 0)).diff(4) * 100.0)
    )
    q["lic_eu_pas_growth_yoy"] = (
        q.sort_values(["geo", "period_str"])
        .groupby("geo")["pas_lic_eu"]
        .transform(lambda s: np.log(s.where(s > 0)).diff(4) * 100.0)
    )
    return q.sort_values(["geo", "period_str"]).reset_index(drop=True)


def build_panel() -> tuple[pd.DataFrame, dict]:
    # Harmonized quarterly backbone + route-shock augmentation.
    base = pd.read_parquet(PROC_DIR / "panel_quarterly_harmonized.parquet").replace([np.inf, -np.inf], np.nan).copy()
    route = pd.read_parquet(PROC_DIR / "panel_quarterly_airport_shocks.parquet").replace([np.inf, -np.inf], np.nan).copy()
    if "period_str" not in base.columns:
        if "period" in base.columns:
            base["period_str"] = base["period"].astype(str)
        else:
            base["period_str"] = base["year"].astype(int).astype(str) + "Q" + base["quarter"].astype(int).astype(str)
    if "period_str" not in route.columns:
        route["period_str"] = route["period"].astype(str)

    # Keep harmonized outcome/mobility fields, merge route-shock terms from specialty panel.
    route_keep = [c for c in route.columns if c not in set(base.columns) or c in {"geo", "period_str"}]
    route_keep = [c for c in route_keep if c in route.columns]
    route_keep = list(dict.fromkeys(["geo", "period_str"] + [c for c in route_keep if c not in {"geo", "period_str"}]))
    panel = base.merge(route[route_keep], on=["geo", "period_str"], how="left")
    panel["hpi_yoy"] = pd.to_numeric(panel.get("hpi_yoy_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(panel.get("hpi_yoy"), errors="coerce")
    )

    hicp_q = build_hicp_airfare_quarterly()
    apal_q = build_airline_quality_quarterly()
    panel = panel.merge(hicp_q.drop(columns=["quarter"], errors="ignore"), on=["geo", "period_str"], how="left")
    panel = panel.merge(apal_q, on=["geo", "period_str"], how="left")

    p = pd.PeriodIndex(panel["period_str"], freq="Q")
    panel["year"] = p.year
    panel["quarter"] = p.quarter
    panel["quarter_id"] = panel["year"].astype(int) * 4 + panel["quarter"].astype(int)
    panel = panel.sort_values(["geo", "quarter_id"]).reset_index(drop=True)
    g = panel.groupby("geo", sort=False)

    lag_cols = [
        "open_rate_norm_q",
        "close_rate_norm_q",
        "open_persist_rate_norm_q",
        "close_persist_rate_norm_q",
        "airfare_yoy_q",
        "airfare_yoy_m_q",
        "lic_neu_share_pas",
        "lic_eu_share_pas",
        "lic_neu_share_move",
        "pax_per_move_total",
        "pax_per_move_lic_neu",
        "pax_per_move_lic_eu",
        "lic_neu_pas_growth_yoy",
        "lic_eu_pas_growth_yoy",
    ]
    for col in lag_cols:
        if col in panel.columns:
            panel[f"L1_{col}"] = g[col].shift(1)

    # Persistence / baseline composition for interactions
    if "lic_neu_share_pas" in panel.columns:
        panel["lic_neu_share_pas_pre2019_mean"] = g["lic_neu_share_pas"].transform(
            lambda s: s.where(panel.loc[s.index, "year"] <= 2019).mean()
        )

    out_pq = PROC_DIR / "panel_quarterly_traveler_quality.parquet"
    out_csv = PROC_DIR / "panel_quarterly_traveler_quality.csv"
    panel.to_parquet(out_pq, index=False)
    panel.to_csv(out_csv, index=False)

    meta = {
        "rows": int(len(panel)),
        "countries": int(panel["geo"].astype(str).nunique()),
        "period_min": str(panel["period_str"].dropna().min()),
        "period_max": str(panel["period_str"].dropna().max()),
        "nonmissing_hpi_yoy": int(panel["hpi_yoy"].notna().sum()) if "hpi_yoy" in panel else 0,
        "nonmissing_airfare_yoy_q": int(panel["airfare_yoy_q"].notna().sum()) if "airfare_yoy_q" in panel else 0,
        "nonmissing_lic_neu_share_pas": int(panel["lic_neu_share_pas"].notna().sum()) if "lic_neu_share_pas" in panel else 0,
        "nonmissing_pax_per_move_total": int(panel["pax_per_move_total"].notna().sum()) if "pax_per_move_total" in panel else 0,
    }
    return panel, meta


def _fit_formula(df_panel: pd.DataFrame, formula: str, label: str, needed: list[str]) -> tuple[pd.DataFrame, str] | None:
    d = df_panel.dropna(subset=needed).copy()
    if len(d) == 0:
        return None
    mod = PanelOLS.from_formula(formula, data=d, drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    coef = pd.DataFrame(
        {
            "model": label,
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.std_errors.values,
            "t_stat": res.tstats.values,
            "p_value": res.pvalues.values,
            "nobs": float(res.nobs),
            "r2_within": float(getattr(res, "rsquared_within", np.nan)),
            "r2_overall": float(getattr(res, "rsquared_overall", np.nan)),
        }
    )
    txt = f"# {label}\nFormula: {formula}\n\n{res.summary}"
    return coef, txt


def estimate_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    df = panel.replace([np.inf, -np.inf], np.nan).copy()
    for col in [
        "hpi_yoy", "L1_air_yoy", "L1_open_rate_norm_q", "L1_close_rate_norm_q",
        "L1_airfare_yoy_q", "L1_lic_neu_share_pas", "L1_pax_per_move_total",
    ]:
        if col in df.columns:
            df[col] = winsorize_series(df[col], 0.01, 0.99)
    if "L1_lic_neu_share_pas" in df.columns:
        df["L1_lic_neu_share_pas_c"] = df["L1_lic_neu_share_pas"] - df["L1_lic_neu_share_pas"].mean(skipna=True)
    if "L1_airfare_yoy_q" in df.columns:
        df["L1_airfare_yoy_q_c"] = df["L1_airfare_yoy_q"] - df["L1_airfare_yoy_q"].mean(skipna=True)

    df_panel = df.set_index(["geo", "quarter_id"]).sort_index()

    route_terms = "L1_open_rate_norm_q + L1_close_rate_norm_q"
    base_need = ["hpi_yoy", "L1_air_yoy", "L1_open_rate_norm_q", "L1_close_rate_norm_q"]
    specs = [
        (
            "quarterly_tq_baseline_matched",
            f"hpi_yoy ~ 1 + L1_air_yoy + {route_terms} + EntityEffects + TimeEffects",
            base_need + ["L1_airfare_yoy_q", "L1_lic_neu_share_pas"],
        ),
        (
            "quarterly_tq_plus_airfare",
            f"hpi_yoy ~ 1 + L1_air_yoy + L1_airfare_yoy_q + {route_terms} + EntityEffects + TimeEffects",
            base_need + ["L1_airfare_yoy_q"],
        ),
        (
            "quarterly_tq_plus_airline_mix",
            f"hpi_yoy ~ 1 + L1_air_yoy + L1_lic_neu_share_pas + L1_pax_per_move_total + {route_terms} + EntityEffects + TimeEffects",
            base_need + ["L1_lic_neu_share_pas", "L1_pax_per_move_total"],
        ),
        (
            "quarterly_tq_full",
            f"hpi_yoy ~ 1 + L1_air_yoy + L1_airfare_yoy_q + L1_lic_neu_share_pas + L1_pax_per_move_total + {route_terms} + EntityEffects + TimeEffects",
            base_need + ["L1_airfare_yoy_q", "L1_lic_neu_share_pas", "L1_pax_per_move_total"],
        ),
        (
            "quarterly_tq_full_interact",
            f"hpi_yoy ~ 1 + L1_air_yoy + L1_airfare_yoy_q_c + L1_lic_neu_share_pas_c + "
            f"L1_air_yoy:L1_lic_neu_share_pas_c + L1_air_yoy:L1_airfare_yoy_q_c + L1_pax_per_move_total + "
            f"{route_terms} + EntityEffects + TimeEffects",
            base_need + ["L1_airfare_yoy_q_c", "L1_lic_neu_share_pas_c", "L1_pax_per_move_total"],
        ),
    ]

    coef_frames, texts = [], []
    for label, formula, needed in specs:
        out = _fit_formula(df_panel, formula, label, needed)
        if out is None:
            continue
        coef, txt = out
        coef_frames.append(coef)
        texts.append(txt)

    coef_df = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()

    stats_vars = [
        "hpi_yoy", "air_yoy", "airfare_yoy_q", "lic_neu_share_pas", "pax_per_move_total", "open_rate_norm_q", "close_rate_norm_q"
    ]
    srows = []
    for c in stats_vars:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        srows.append(
            {
                "variable": c,
                "n": int(x.notna().sum()),
                "mean": float(x.mean()) if x.notna().any() else np.nan,
                "std": float(x.std()) if x.notna().any() else np.nan,
                "p25": float(x.quantile(0.25)) if x.notna().any() else np.nan,
                "p50": float(x.quantile(0.50)) if x.notna().any() else np.nan,
                "p75": float(x.quantile(0.75)) if x.notna().any() else np.nan,
            }
        )
    stats_df = pd.DataFrame(srows)
    return coef_df, texts, stats_df


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def _coef_entry(coef_df: pd.DataFrame, model: str, term: str) -> tuple[str, str]:
    r = coef_df[(coef_df["model"] == model) & (coef_df["term"] == term)]
    if r.empty:
        return "", ""
    x = r.iloc[0]
    return f"{x['coef']:.3f}{_stars(float(x['p_value']))}", f"({x['std_err']:.3f})"


def _plot_coef_figure(coef_df: pd.DataFrame) -> None:
    if not PAPER_FIGS_DIR.exists():
        return
    keep = [
        ("quarterly_tq_baseline_matched", "L1_air_yoy", "Air growth (baseline matched)"),
        ("quarterly_tq_plus_airfare", "L1_air_yoy", "Air growth (+ airfare)"),
        ("quarterly_tq_plus_airfare", "L1_airfare_yoy_q", "Airfare inflation"),
        ("quarterly_tq_plus_airline_mix", "L1_lic_neu_share_pas", "Non-EU carrier share"),
        ("quarterly_tq_plus_airline_mix", "L1_pax_per_move_total", "Passengers per movement"),
        ("quarterly_tq_full", "L1_airfare_yoy_q", "Airfare inflation (full)"),
        ("quarterly_tq_full", "L1_lic_neu_share_pas", "Non-EU carrier share (full)"),
        ("quarterly_tq_full_interact", "L1_air_yoy:L1_airfare_yoy_q_c", "Air growth × airfare"),
        ("quarterly_tq_full_interact", "L1_air_yoy:L1_lic_neu_share_pas_c", "Air growth × non-EU share"),
    ]
    rows = []
    for m, t, lbl in keep:
        r = coef_df[(coef_df["model"] == m) & (coef_df["term"] == t)]
        if r.empty:
            continue
        x = r.iloc[0]
        rows.append({"label": lbl, "coef": float(x["coef"]), "se": float(x["std_err"])})
    if not rows:
        return
    d = pd.DataFrame(rows)
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.axvline(0, color="#444", lw=1, ls="--")
    ax.errorbar(d["coef"], y, xerr=1.96 * d["se"], fmt="none", ecolor="#333", elinewidth=1.4, capsize=3)
    colors = ["#1f4e79" if "Airfare" in l else "#b22222" if "non-EU" in l else "#2f7d32" for l in d["label"]]
    ax.scatter(d["coef"], y, c=colors, s=44, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=9)
    ax.set_xlabel("Coefficient estimate (95% CI)")
    ax.set_title("Traveler-quality proxies in quarterly house-price FE models")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PAPER_FIGS_DIR / "fig_traveler_quality_coefficients.pdf")
    plt.close(fig)


def _plot_time_series(panel: pd.DataFrame) -> None:
    if not PAPER_FIGS_DIR.exists():
        return
    df = panel.copy()
    keep_cols = ["period_str", "hpi_yoy", "air_yoy", "airfare_yoy_q", "lic_neu_share_pas"]
    if any(c not in df.columns for c in ["period_str", "hpi_yoy"]):
        return
    d = df[keep_cols].dropna(subset=["period_str"]).copy()
    d["period"] = pd.PeriodIndex(d["period_str"], freq="Q")
    agg = d.groupby("period", as_index=False)[[c for c in keep_cols if c != "period_str"]].mean(numeric_only=True)
    if agg.empty:
        return
    for col in ["hpi_yoy", "air_yoy", "airfare_yoy_q", "lic_neu_share_pas"]:
        if col not in agg:
            continue
        x = pd.to_numeric(agg[col], errors="coerce")
        mu, sd = x.mean(skipna=True), x.std(skipna=True)
        agg[f"z_{col}"] = (x - mu) / (sd if (pd.notna(sd) and sd != 0) else 1.0)
    fig, ax = plt.subplots(figsize=(10.7, 4.6))
    x = agg["period"].astype(str)
    if "z_hpi_yoy" in agg:
        ax.plot(x, agg["z_hpi_yoy"], label="HPI YoY", lw=2.1, color="#1f4e79")
    if "z_air_yoy" in agg:
        ax.plot(x, agg["z_air_yoy"], label="Air passenger YoY", lw=1.8, color="#2f7d32")
    if "z_airfare_yoy_q" in agg:
        ax.plot(x, agg["z_airfare_yoy_q"], label="Airfare inflation (CP0733)", lw=1.8, color="#b22222")
    if "z_lic_neu_share_pas" in agg:
        ax.plot(x, agg["z_lic_neu_share_pas"], label="Non-EU carrier share", lw=1.8, color="#946c00")
    ticks = np.linspace(0, len(x) - 1, min(10, len(x))).astype(int) if len(x) > 0 else []
    ax.set_xticks(ticks)
    ax.set_xticklabels([x.iloc[i] for i in ticks], rotation=0, fontsize=8)
    ax.axhline(0, color="#777", lw=0.8, alpha=0.6)
    ax.grid(alpha=0.2)
    ax.set_ylabel("Standardized units")
    ax.set_title("Quarterly traveler-quality proxies and housing-price dynamics (country averages)")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(PAPER_FIGS_DIR / "fig_traveler_quality_timeseries.pdf")
    plt.close(fig)


def _write_paper_table(coef_df: pd.DataFrame) -> None:
    if not PAPER_TABLES_DIR.exists():
        return
    models = [
        ("quarterly_tq_baseline_matched", "Matched baseline"),
        ("quarterly_tq_plus_airfare", "+ Airfare"),
        ("quarterly_tq_full", "Full TQ"),
        ("quarterly_tq_full_interact", "Full TQ + int."),
    ]
    rows = [
        ("L1_air_yoy", "Lagged air-passenger YoY growth"),
        ("L1_airfare_yoy_q", "Lagged airfare inflation (HICP CP0733, YoY)"),
        ("L1_lic_neu_share_pas", "Lagged non-EU carrier passenger share"),
        ("L1_pax_per_move_total", "Lagged passengers per commercial movement"),
        ("L1_air_yoy:L1_airfare_yoy_q_c", "Lagged air growth × demeaned airfare inflation"),
        ("L1_air_yoy:L1_lic_neu_share_pas_c", "Lagged air growth × demeaned non-EU carrier share"),
        ("L1_open_rate_norm_q", "Lagged route opening rate"),
        ("L1_close_rate_norm_q", "Lagged route closure rate"),
    ]
    colspec = "p{5.9cm}" + "".join(["p{2.15cm}"] * len(models))
    out = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Traveler-quality proxies in quarterly country FE models}",
        r"\label{tab:traveler_quality_quarterly}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & " + " & ".join(lbl for _, lbl in models) + r" \\",
        r"\midrule",
    ]
    for term, label in rows:
        coef_line, se_line = [label], [""]
        for m, _ in models:
            c, s = _coef_entry(coef_df, m, term)
            coef_line.append(c)
            se_line.append(s)
        out.append(" & ".join(coef_line) + r" \\")
        out.append(" & ".join(se_line) + r" \\")
    out.append(r"\midrule")
    for stat in ["Country FE", "Quarter FE", "Two-way clustered SE", "Observations"]:
        row = [stat]
        for m, _ in models:
            sub = coef_df[coef_df["model"] == m]
            if sub.empty:
                row.append("")
            elif stat == "Observations":
                row.append(str(int(sub["nobs"].max())))
            else:
                row.append("Yes")
        out.append(" & ".join(row) + r" \\")
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Notes: Dependent variable is quarterly house-price YoY growth. All models include country and quarter fixed effects and two-way clustered standard errors (country and quarter). Airfare inflation is based on Eurostat HICP \texttt{CP0733} (air passenger transport). Carrier-mix proxies are aggregated from airport-level \texttt{AVIA\_TF\_APAL} data using airline licence categories (\texttt{LIC\_EU} vs \texttt{LIC\_NEU}).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (PAPER_TABLES_DIR / "tab_traveler_quality_quarterly.tex").write_text("\n".join(out))


def write_paper_assets(panel: pd.DataFrame, coef_df: pd.DataFrame) -> None:
    if not PAPER_DIR.exists():
        return
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    _write_paper_table(coef_df)
    _plot_coef_figure(coef_df)
    _plot_time_series(panel)


def main() -> None:
    for d in [PROC_DIR, RESULTS_DIR, META_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    panel, meta = build_panel()
    coef_df, texts, stats_df = estimate_models(panel)

    coef_path = RESULTS_DIR / "traveler_quality_coefficients.csv"
    txt_path = RESULTS_DIR / "traveler_quality_summaries.txt"
    stats_path = RESULTS_DIR / "traveler_quality_sample_stats.csv"
    meta_path = META_DIR / "traveler_quality_summary.json"
    coef_df.to_csv(coef_path, index=False)
    txt_path.write_text("\n\n".join(texts))
    stats_df.to_csv(stats_path, index=False)
    if not coef_df.empty:
        meta["models"] = (
            coef_df.groupby("model", as_index=False)["nobs"].max().rename(columns={"nobs": "nobs_model"}).to_dict(orient="records")
        )
    meta_path.write_text(json.dumps(meta, indent=2))

    write_paper_assets(panel, coef_df)

    print(f"[ok] wrote {coef_path}")
    print(f"[ok] wrote {txt_path}")
    print(f"[ok] wrote {stats_path}")
    print(f"[ok] wrote {meta_path}")


if __name__ == "__main__":
    main()
