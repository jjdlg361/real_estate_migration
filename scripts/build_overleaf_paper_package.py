#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from linearmodels.iv import IV2SLS
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "metadata"
RESULTS_DIR = ROOT / "results"
PAPER_DIR = ROOT / "paper_overleaf"
FIG_DIR = PAPER_DIR / "figures"
TAB_DIR = PAPER_DIR / "tables"
ZIP_PATH = ROOT / "real_estate_migration_overleaf_upload.zip"


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _iter_exterior_coords(geom) -> list[np.ndarray]:
    polys: list[Polygon] = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        return []
    out = []
    for p in polys:
        if p.is_empty:
            continue
        out.append(np.asarray(p.exterior.coords))
    return out


def load_country_polygons() -> dict[str, object]:
    path = DATA_DIR / "raw" / "gisco_nuts2_2021_4326.geojson"
    if not path.exists():
        return {}
    gj = json.loads(path.read_text())
    by_country: dict[str, list[object]] = {}
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        cn = props.get("CNTR_CODE")
        if not cn:
            continue
        by_country.setdefault(str(cn), []).append(shape(feat["geometry"]))
    dissolved = {k: unary_union(v) for k, v in by_country.items() if v}
    return dissolved


def _draw_country_choropleth(ax, geoms: dict[str, object], values: pd.Series, title: str, cmap: str = "viridis", diverging: bool = False) -> None:
    vals = values.dropna()
    countries = [c for c in vals.index if c in geoms]
    if not countries:
        ax.set_axis_off()
        ax.set_title(title)
        return
    v = vals.loc[countries]
    if diverging:
        vmax = float(np.nanmax(np.abs(v.values))) if len(v) else 1.0
        vmin = -vmax
    else:
        vmin, vmax = float(np.nanmin(v.values)), float(np.nanmax(v.values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6

    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    patches = []
    colors = []
    for c in countries:
        geom = geoms[c]
        for coords in _iter_exterior_coords(geom):
            patches.append(MplPolygon(coords[:, :2], closed=True))
            colors.append(vals.loc[c])
    if patches:
        pc = PatchCollection(patches, cmap=cmap_obj, norm=norm, linewidths=0.35, edgecolor="#FFFFFF")
        pc.set_array(np.asarray(colors))
        ax.add_collection(pc)
        cbar = plt.colorbar(pc, ax=ax, fraction=0.040, pad=0.01)
        cbar.ax.tick_params(labelsize=7)

    # draw missing countries in very light grey for context
    missing = [c for c in geoms if c not in countries]
    miss_patches = []
    for c in missing:
        for coords in _iter_exterior_coords(geoms[c]):
            miss_patches.append(MplPolygon(coords[:, :2], closed=True))
    if miss_patches:
        pc_m = PatchCollection(miss_patches, facecolor="#F1F1F1", edgecolor="#DDDDDD", linewidths=0.25)
        ax.add_collection(pc_m)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-25, 40)
    ax.set_ylim(34, 72)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def _draw_country_borders(ax, geoms: dict[str, object], facecolor: str = "#F7F7F7", edgecolor: str = "#CFCFCF") -> None:
    patches = []
    for geom in geoms.values():
        for coords in _iter_exterior_coords(geom):
            patches.append(MplPolygon(coords[:, :2], closed=True))
    if patches:
        pc = PatchCollection(patches, facecolor=facecolor, edgecolor=edgecolor, linewidths=0.4, zorder=0)
        ax.add_collection(pc)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-25, 40)
    ax.set_ylim(34, 72)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def fmt_num(x: float | int | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.{digits}f}"


def fmt_int(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{int(round(float(x))):,}"


def fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def stars(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def latex_escape(s: str) -> str:
    out = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def coef_lookup(df: pd.DataFrame, model: str, term: str) -> dict | None:
    x = df[(df["model"] == model) & (df["term"] == term)]
    if x.empty:
        return None
    return x.iloc[0].to_dict()


def coef_cell(df: pd.DataFrame, model: str, term: str) -> tuple[str, str]:
    row = coef_lookup(df, model, term)
    if row is None:
        return "", ""
    est = f"{row['coef']:.3f}{stars(row.get('p_value'))}"
    se = row.get("std_err")
    se_txt = "" if pd.isna(se) else f"({se:.3f})"
    return est, se_txt


def build_table_coverage() -> str:
    panel = load_json(META_DIR / "panel_build_summary.json")
    ss = load_json(META_DIR / "shiftshare_iv_summary.json")
    fs = load_json(META_DIR / "flight_shock_build_summary.json")
    cross = load_json(META_DIR / "airport_nuts2_crosswalk_summary.json")
    reg_a = load_json(META_DIR / "regional_panel_summary.json")
    reg_q = load_json(META_DIR / "regional_route_quarterly_panel_summary.json")

    rows = [
        ("National annual panel", fmt_int(panel["annual_rows"]), fmt_int(panel["annual_countries"]), f"{panel['annual_year_min']}--{panel['annual_year_max']}", "Eurostat HPI + migration + air + macro controls"),
        ("National quarterly panel", fmt_int(panel["quarterly_rows"]), fmt_int(panel["quarterly_countries"]), f"{panel['quarterly_period_min']}--{panel['quarterly_period_max']}", "Eurostat HPI + air passengers"),
        ("Annual baseline estimation sample", fmt_int(panel["annual_full_sample_rows_for_baseline"]), "", "2014--2024", "Non-missing HPI growth and lagged migration"),
        ("Quarterly baseline estimation sample", fmt_int(panel["quarterly_full_sample_rows_for_baseline"]), "", "2006Q1--2025Q3", "Non-missing HPI YoY and lagged air YoY"),
        ("OD migration IV panel", fmt_int(ss["iv_rows"]), fmt_int(ss["od_destinations"]), f"{ss['od_year_min']}--{ss['od_year_max']}", "Destination-origin immigration exposures"),
        ("Monthly route-month panel", fmt_int(fs["route_month_rows"]), fmt_int(fs["countries"]), "2004--2025", "Airport-partner route activity"),
        ("Country-quarter route shocks", fmt_int(fs["country_quarter_rows"]), fmt_int(fs["countries"]), f"{fs['country_quarter_min']}--{fs['country_quarter_max']}", "Aggregated route openings/closures"),
        ("Airport-to-NUTS2 crosswalk", fmt_int(cross["airports_total"]), "", "", f"Mapped {cross['nuts2_matched']}/{cross['airports_total']} airports ({100*cross['match_rate_nuts2']:.1f}%)"),
        ("NUTS2 annual panel (corrected RHPI)", fmt_int(reg_a["rows"]), fmt_int(reg_a["regions"]), f"{reg_a['year_min']}--{reg_a['year_max']}", f"{reg_a['countries']} countries after RHPI vintage filter"),
        ("NUTS2 quarterly route-shock panel (corrected RHPI)", fmt_int(reg_q["rows"]), fmt_int(reg_q["regions"]), f"{reg_q['period_min']}--{reg_q['period_max']}", f"{reg_q['countries']} countries after RHPI vintage filter"),
    ]
    body = "\n".join(
        f"{latex_escape(a)} & {b} & {c} & {latex_escape(d)} & {latex_escape(e)} \\\\"
        for a, b, c, d, e in rows
    )
    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{Dataset architecture and usable coverage}}
\label{{tab:coverage}}
\begin{{threeparttable}}
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{p{{4.0cm}}cccc}}
\toprule
Dataset block & Rows & Units & Period & Notes \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item Notes: Regional OECD RHPI panels are filtered to total housing stock (VINTAGE = \texttt{{\_T}}) and non-seasonally adjusted observations (ADJUSTMENT = \texttt{{N}}) to avoid duplicate region-period observations.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip()


def build_table_descriptive() -> str:
    stats = pd.read_csv(RESULTS_DIR / "sample_stats.csv")
    keep_order = [
        ("annual", "hpi_growth", "Annual HPI growth (%)"),
        ("annual", "net_migration_rate", "Net migration rate (per 1,000)"),
        ("annual", "air_growth", "Air passenger growth (%)"),
        ("annual", "gdp_pc_growth", "GDP per capita growth (%)"),
        ("annual", "unemployment_rate", "Unemployment rate (%)"),
        ("annual", "inflation_hicp", "HICP inflation (%)"),
        ("quarterly", "hpi_yoy", "Quarterly HPI YoY growth (%)"),
        ("quarterly", "air_yoy", "Quarterly air passenger YoY growth (%)"),
        ("quarterly", "dlog_hpi_qoq", "Quarterly HPI QoQ log change (pp)"),
    ]
    rows = []
    for panel, var, label in keep_order:
        x = stats[(stats["panel"] == panel) & (stats["variable"] == var)]
        if x.empty:
            continue
        r = x.iloc[0]
        rows.append(
            (
                label,
                fmt_int(r["n"]),
                fmt_num(r["mean"], 2),
                fmt_num(r["std"], 2),
                fmt_num(r["p10"], 2),
                fmt_num(r["p50"], 2),
                fmt_num(r["p90"], 2),
            )
        )
    body = "\n".join(
        f"{latex_escape(a)} & {b} & {c} & {d} & {e} & {f} & {g} \\\\"
        for a, b, c, d, e, f, g in rows
    )
    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{Descriptive statistics for main national analysis variables}}
\label{{tab:desc}}
\begin{{threeparttable}}
\small
\begin{{tabular}}{{p{{5.5cm}}rrrrrr}}
\toprule
Variable & N & Mean & SD & P10 & Median & P90 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item Notes: Statistics are computed from the processed annual and quarterly panels. Units follow the source definitions and transformations described in Section~\ref{{sec:data}}.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip()


def build_table_baseline_annual() -> str:
    coef = pd.read_csv(RESULTS_DIR / "model_coefficients.csv")
    models = [
        ("annual_fe_migration", "Migration only"),
        ("annual_fe_migration_flights", "+ Flights"),
        ("annual_fe_full_controls", "+ Full controls"),
    ]
    rows_spec = [
        ("L1_net_migration_rate", "Lagged net migration rate"),
        ("L1_air_growth", "Lagged air-passenger growth"),
        ("L1_gdp_pc_growth", "Lagged GDP per-capita growth"),
        ("L1_unemployment_rate", "Lagged unemployment rate"),
        ("L1_inflation_hicp", "Lagged inflation"),
        ("L1_long_rate", "Lagged long-term rate"),
        ("L1_pop_growth", "Lagged population growth"),
    ]

    lines = []
    for term, label in rows_spec:
        ests, ses = [], []
        for m, _ in models:
            e, s = coef_cell(coef, m, term)
            ests.append(e)
            ses.append(s)
        lines.append(f"{latex_escape(label)} & " + " & ".join(ests) + r" \\")
        lines.append(" & " + " & ".join(ses) + r" \\")
    lines.append(r"\midrule")
    lines.append("Country FE & Yes & Yes & Yes \\\\")
    lines.append("Year FE & Yes & Yes & Yes \\\\")
    lines.append("Clustered SE (country) & Yes & Yes & Yes \\\\")
    lines.append("Controls included & No & No & Yes \\\\")
    nobs = [coef[coef["model"] == m]["nobs"].dropna().iloc[0] for m, _ in models]
    r2 = [coef[coef["model"] == m]["r2_within"].dropna().iloc[0] for m, _ in models]
    lines.append("Observations & " + " & ".join(fmt_int(x) for x in nobs) + r" \\")
    lines.append(r"$R^2$ (within) & " + " & ".join(fmt_num(x, 3) for x in r2) + r" \\")

    header_models = " & ".join(latex_escape(lbl) for _, lbl in models)
    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{Baseline annual two-way fixed-effects estimates (national panel)}}
\label{{tab:baseline_annual}}
\begin{{threeparttable}}
\small
\begin{{tabular}}{{p{{5.4cm}}ccc}}
\toprule
 & {header_models} \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item Notes: Dependent variable is annual house-price growth. All regressors are lagged one period. Standard errors (in parentheses) are clustered at the country level. Significance: *** p<0.01, ** p<0.05, * p<0.10.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip()


def build_table_quarterly() -> str:
    base = pd.read_csv(RESULTS_DIR / "model_coefficients.csv")
    adv = pd.read_csv(RESULTS_DIR / "advanced_model_coefficients.csv")
    models = [
        ("quarterly_fe_air_yoy", "Air YoY (1 lag)", base, "Baseline"),
        ("quarterly_fe_air_yoy_lags", "Air YoY (2 lags)", base, "Baseline"),
        ("quarterly_fe_airplus_routecounts", "+ Route counts", adv, "Route-shock FE"),
        ("quarterly_fe_airplus_route_rates", "+ Route rates", adv, "Route-shock FE"),
    ]
    row_specs = [
        ("L1_air_yoy", "Lagged air-passenger YoY"),
        ("L2_air_yoy", "2nd lag air-passenger YoY"),
        ("L1_net_openings_q", "Lagged net route openings (count)"),
        ("L1_open_rate_norm_q", "Lagged persistent opening intensity"),
        ("L1_close_rate_norm_q", "Lagged persistent closure intensity"),
    ]
    lines = []
    for term, label in row_specs:
        ests, ses = [], []
        for m, _, df, _ in models:
            e, s = coef_cell(df, m, term)
            ests.append(e)
            ses.append(s)
        lines.append(f"{latex_escape(label)} & " + " & ".join(ests) + r" \\")
        lines.append(" & " + " & ".join(ses) + r" \\")
    lines.append(r"\midrule")
    lines.append("Country FE & Yes & Yes & Yes & Yes \\\\")
    lines.append("Quarter FE & Yes & Yes & Yes & Yes \\\\")
    lines.append("Two-way clustered SE & No & No & Yes & Yes \\\\")
    nobs = []
    for m, _, df, _ in models:
        x = df[df["model"] == m]
        nobs.append(x["nobs"].dropna().iloc[0] if not x.empty else np.nan)
    lines.append("Observations & " + " & ".join(fmt_int(x) for x in nobs) + r" \\")
    header_models = " & ".join(latex_escape(lbl) for _, lbl, _, _ in models)
    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{Quarterly national fixed-effects estimates: air mobility and route-shock extensions}}
\label{{tab:quarterly_models}}
\begin{{threeparttable}}
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{p{{5.8cm}}cccc}}
\toprule
 & {header_models} \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item Notes: Dependent variable is quarterly house-price YoY growth. Baseline models use country-clustered standard errors; route-shock FE models use two-way clustering by country and quarter. Persistent route-event intensity variables come from monthly airport-partner route shocks aggregated to the country-quarter level.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip()


def build_table_iv_results() -> str:
    coef = pd.read_csv(RESULTS_DIR / "advanced_model_coefficients.csv")
    models = [("country_iv_fe", "Composite shift-share IV"), ("country_iv_fe_asylum", "Asylum shift-share IV")]
    row_specs = [
        ("L1_net_migration_rate", "Lagged net migration rate (endogenous)"),
        ("L1_air_growth", "Lagged air-passenger growth"),
        ("L1_gdp_pc_growth", "Lagged GDP per-capita growth"),
        ("L1_unemployment_rate", "Lagged unemployment rate"),
    ]
    lines = []
    for term, label in row_specs:
        ests, ses = [], []
        for m, _ in models:
            e, s = coef_cell(coef, m, term)
            ests.append(e)
            ses.append(s)
        lines.append(f"{latex_escape(label)} & " + " & ".join(ests) + r" \\")
        lines.append(" & " + " & ".join(ses) + r" \\")
    lines.append(r"\midrule")
    lines.append("Country FE & Yes & Yes \\\\")
    lines.append("Year FE & Yes & Yes \\\\")
    lines.append("Clustered SE (country) & Yes & Yes \\\\")
    nobs = [coef[coef["model"] == m]["nobs"].dropna().iloc[0] for m, _ in models]
    lines.append("Observations & " + " & ".join(fmt_int(x) for x in nobs) + r" \\")
    header_models = " & ".join(latex_escape(lbl) for _, lbl in models)
    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{Country-year IV estimates with country and year fixed effects}}
\label{{tab:iv_results}}
\begin{{threeparttable}}
\small
\begin{{tabular}}{{p{{6.0cm}}cc}}
\toprule
 & {header_models} \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item Notes: Dependent variable is annual house-price growth. The endogenous regressor is lagged net migration rate. Composite IV uses a World Bank push-shock shift-share exposure plus a leave-one-out origin-supply component; asylum IV uses a leave-one-out asylum-applications shift-share exposure.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip()


def _fit_iv_for_diagnostics(instruments: list[str]) -> tuple[object, pd.DataFrame]:
    d = pd.read_parquet(PROC_DIR / "panel_annual_iv.parquet").replace([np.inf, -np.inf], np.nan).copy()
    h = pd.read_parquet(PROC_DIR / "panel_annual_harmonized.parquet").replace([np.inf, -np.inf], np.nan).copy()
    harmonized_cols = [
        "hpi_growth_harmonized",
        "net_migration_rate_harmonized",
        "air_growth_harmonized",
        "gdp_pc_growth_harmonized",
        "unemployment_rate_harmonized",
        "inflation_hicp_harmonized",
        "long_rate_harmonized",
        "pop_growth_harmonized",
    ]
    missing_h_cols = [c for c in harmonized_cols if c not in d.columns]
    if missing_h_cols:
        h = h[["geo", "year"] + missing_h_cols].drop_duplicates(["geo", "year"])
        d = d.merge(h, on=["geo", "year"], how="left")
    d["hpi_growth"] = pd.to_numeric(d.get("hpi_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d.get("hpi_growth"), errors="coerce")
    )
    d["net_migration_rate"] = pd.to_numeric(d.get("net_migration_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d["net_migration_rate"], errors="coerce")
    )
    d["air_growth"] = pd.to_numeric(d.get("air_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d.get("air_growth"), errors="coerce")
    )
    d["gdp_pc_growth"] = pd.to_numeric(d.get("gdp_pc_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d["gdp_pc_growth"], errors="coerce")
    )
    d["unemployment_rate"] = pd.to_numeric(d.get("unemployment_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d["unemployment_rate"], errors="coerce")
    )
    d["inflation_hicp"] = pd.to_numeric(d.get("inflation_hicp_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d["inflation_hicp"], errors="coerce")
    )
    d["long_rate"] = pd.to_numeric(d.get("long_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d.get("long_rate"), errors="coerce")
    )
    d["pop_growth"] = pd.to_numeric(d.get("pop_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d.get("pop_growth"), errors="coerce")
    )
    d = d.sort_values(["geo", "year"]).reset_index(drop=True)
    g = d.groupby("geo", sort=False)
    for c in [
        "net_migration_rate",
        "air_growth",
        "gdp_pc_growth",
        "unemployment_rate",
        "inflation_hicp",
        "long_rate",
        "pop_growth",
    ]:
        d[f"L1_{c}"] = g[c].shift(1)
    keep = [
        "geo",
        "year",
        "hpi_growth",
        "L1_net_migration_rate",
        "L1_air_growth",
        "L1_gdp_pc_growth",
        "L1_unemployment_rate",
        "L1_inflation_hicp",
        "L1_long_rate",
        "L1_pop_growth",
    ] + instruments
    d = d[keep].copy()
    for col in ["hpi_growth", "L1_air_growth", "L1_ss_loo_origin_supply_logdiff"]:
        if col in d.columns:
            x = pd.to_numeric(d[col], errors="coerce")
            lo, hi = x.quantile([0.01, 0.99])
            d[col] = x.clip(lo, hi)
    needed = [
        "hpi_growth",
        "L1_net_migration_rate",
        "L1_air_growth",
        "L1_gdp_pc_growth",
        "L1_unemployment_rate",
        "L1_inflation_hicp",
        "L1_long_rate",
        "L1_pop_growth",
    ] + instruments
    d = d.dropna(subset=needed).copy()
    formula = (
        "hpi_growth ~ 1 + L1_air_growth + L1_gdp_pc_growth + L1_unemployment_rate + "
        "L1_inflation_hicp + L1_long_rate + L1_pop_growth + C(geo) + C(year) "
        f"[L1_net_migration_rate ~ {' + '.join(instruments)}]"
    )
    res = IV2SLS.from_formula(formula, data=d).fit(cov_type="clustered", clusters=d["geo"])
    return res, d


def build_table_iv_diagnostics() -> str:
    rows = []

    specs = [
        ("Composite shift-share IV", ["L1_ss_push_index_wb", "L1_ss_loo_origin_supply_logdiff"], True),
        ("Asylum shift-share IV", ["L1_ss_asylum_loo_logdiff"], False),
    ]
    for label, instrs, has_overid in specs:
        res, d = _fit_iv_for_diagnostics(instrs)
        diag = res.first_stage.diagnostics.iloc[0]
        sargan_stat = sargan_p = basmann_stat = basmann_p = None
        if has_overid:
            sarg = getattr(res, "sargan", None)
            bas = getattr(res, "basmann", None)
            if sarg is not None:
                sargan_stat, sargan_p = float(sarg.stat), float(sarg.pval)
            if bas is not None:
                basmann_stat, basmann_p = float(bas.stat), float(bas.pval)
        rows.append(
            (
                label,
                len(instrs),
                float(diag.get("partial.rsquared", np.nan)),
                float(diag.get("f.stat", np.nan)),
                float(diag.get("f.pval", np.nan)),
                sargan_stat,
                sargan_p,
                basmann_stat,
                basmann_p,
                len(d),
            )
        )

    body = "\n".join(
        f"{latex_escape(lbl)} & {n_instr} & {fmt_num(pr2,3)} & {fmt_num(fstat,3)} & {fmt_p(fp)} & "
        f"{fmt_num(ss,3)} & {fmt_p(sp)} & {fmt_num(bs,3)} & {fmt_p(bp)} & {fmt_int(n)} \\\\"
        for lbl, n_instr, pr2, fstat, fp, ss, sp, bs, bp, n in rows
    )

    return rf"""
\begin{{table}}[!htbp]
\centering
\caption{{IV first-stage and over-identification diagnostics}}
\label{{tab:iv_diag}}
\begin{{threeparttable}}
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{p{{4.2cm}}rrrrrrrrr}}
\toprule
Specification & \# IVs & Partial $R^2$ & Partial F & F p-val & Sargan & Sargan p & Basmann & Basmann p & N \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item Notes: Diagnostics are extracted from the same 2SLS specifications used in Table~\ref{{tab:iv_results}} with country and year fixed effects and country-clustered covariance. Over-identification tests are only defined for the overidentified (two-instrument) specification.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip()


def build_table_regional_and_event() -> tuple[str, str]:
    adv = pd.read_csv(RESULTS_DIR / "advanced_model_coefficients.csv")
    es_sum = pd.read_json(RESULTS_DIR / "event_study_pretrend_summary.json")

    # Regional FE table
    models = [
        ("nuts2_twfe_baseline", "NUTS2 TWFE baseline"),
        ("nuts2_twfe_post2020_interaction", "NUTS2 TWFE + post-2020 interactions"),
    ]
    row_specs = [
        ("L1_net_migration_rate", "Lagged net migration rate"),
        ("L1_air_growth", "Lagged air-passenger growth"),
        ("L1_unemployment_rate", "Lagged unemployment rate"),
        ("L1_gdp_pc_growth", "Lagged GDP per-capita growth"),
        ("L1_net_migration_rate:post_2020", "Lagged migration × post-2020"),
        ("L1_air_growth:post_2020", "Lagged air growth × post-2020"),
    ]
    lines = []
    for term, label in row_specs:
        ests, ses = [], []
        for m, _ in models:
            e, s = coef_cell(adv, m, term)
            ests.append(e)
            ses.append(s)
        lines.append(f"{latex_escape(label)} & " + " & ".join(ests) + r" \\")
        lines.append(" & " + " & ".join(ses) + r" \\")
    lines.append(r"\midrule")
    lines.append("Region FE & Yes & Yes \\\\")
    lines.append("Year FE & Yes & Yes \\\\")
    lines.append("Two-way clustered SE & Yes & Yes \\\\")
    lines.append("Observations & " + " & ".join(fmt_int(adv[adv['model'] == m]['nobs'].iloc[0]) for m, _ in models) + r" \\")
    regional_table = rf"""
\begin{{table}}[!htbp]
\centering
\caption{{Exploratory NUTS2 annual fixed-effects estimates (corrected RHPI sample)}}
\label{{tab:regional_fe}}
\begin{{threeparttable}}
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{p{{6.0cm}}cc}}
\toprule
 & {latex_escape(models[0][1])} & {latex_escape(models[1][1])} \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item Notes: Dependent variable is OECD regional RHPI annual growth. Sample shrinks materially after removing duplicate RHPI vintages and keeping total-stock (\texttt{{VINTAGE=\_T}}) observations only. One coefficient in the baseline model has an undefined clustered standard error due the small number of country clusters.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip()

    # Event-study diagnostics table
    map_label = {
        "regional_event_study_open_persistent": "Persistent route opening event",
        "regional_event_study_close_persistent": "Persistent route closure event",
    }
    rows = []
    for r in es_sum.to_dict(orient="records"):
        ptest = r.get("pretrend_joint_test", {}) or {}
        rows.append(
            (
                map_label.get(r["model"], r["model"]),
                r.get("regions_total"),
                r.get("treated_regions"),
                r.get("nobs"),
                r.get("event_threshold_rate"),
                ptest.get("df"),
                ptest.get("stat"),
                ptest.get("p_value"),
            )
        )
    body = "\n".join(
        f"{latex_escape(lbl)} & {fmt_int(rt)} & {fmt_int(tr)} & {fmt_int(n)} & {fmt_num(thr,3)} & {fmt_int(df)} & {fmt_num(stat,3)} & {fmt_p(p)} \\\\"
        for lbl, rt, tr, n, thr, df, stat, p in rows
    )
    event_table = rf"""
\begin{{table}}[!htbp]
\centering
\caption{{Regional event-study pre-trend diagnostics (persistent route events)}}
\label{{tab:event_pretrend}}
\begin{{threeparttable}}
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{p{{5.6cm}}rrrrrrr}}
\toprule
Event definition & Regions & Treated & Obs. & Threshold & Lead df & $\chi^2$ lead test & p-value \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item Notes: Event-study models use the first qualifying event per region, a symmetric event window of eight quarters before and after, omit event time $-1$, and include region and quarter fixed effects with two-way clustered standard errors.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip()

    return regional_table, event_table


def plot_national_trends() -> None:
    ann = pd.read_parquet(PROC_DIR / "panel_annual_harmonized.parquet").copy()
    q = pd.read_parquet(PROC_DIR / "panel_quarterly_harmonized.parquet").copy()

    ann["hpi_growth"] = pd.to_numeric(ann.get("hpi_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(ann.get("hpi_growth"), errors="coerce")
    )
    ann["net_migration_rate"] = pd.to_numeric(ann.get("net_migration_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(ann.get("net_migration_rate"), errors="coerce")
    )
    ann["air_growth"] = pd.to_numeric(ann.get("air_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(ann.get("air_growth"), errors="coerce")
    )
    q["hpi_yoy"] = pd.to_numeric(q.get("hpi_yoy_harmonized"), errors="coerce").combine_first(pd.to_numeric(q.get("hpi_yoy"), errors="coerce"))

    ann = ann.sort_values(["year", "geo"])
    ann_s = (
        ann.groupby("year")
        .agg(
            hpi_growth=("hpi_growth", "mean"),
            net_migration_rate=("net_migration_rate", "mean"),
            air_growth=("air_growth", "mean"),
        )
        .reset_index()
    )
    ann_s = ann_s[(ann_s["year"] >= 2004) & (ann_s["year"] <= 2024)].copy()
    for c in ["hpi_growth", "net_migration_rate", "air_growth"]:
        x = ann_s[c]
        ann_s[f"z_{c}"] = (x - x.mean()) / x.std(ddof=0)

    q["period"] = pd.PeriodIndex(q["period_str"].astype(str), freq="Q")
    q_s = q.groupby("period").agg(hpi_yoy=("hpi_yoy", "mean"), air_yoy=("air_yoy", "mean")).reset_index()
    q_s = q_s[(q_s["period"] >= pd.Period("2004Q1")) & (q_s["period"] <= pd.Period("2025Q3"))].copy()
    for c in ["hpi_yoy", "air_yoy"]:
        x = q_s[c]
        q_s[f"z_{c}"] = (x - x.mean()) / x.std(ddof=0)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(ann_s["year"], ann_s["z_hpi_growth"], lw=2.2, color="#0B4F6C", label="HPI growth (annual)")
    ax.plot(ann_s["year"], ann_s["z_net_migration_rate"], lw=2.0, color="#B22222", label="Net migration rate")
    ax.plot(ann_s["year"], ann_s["z_air_growth"], lw=2.0, color="#2A9D8F", label="Air passenger growth")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("National annual averages (standardized)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Z-score")
    ax.legend(loc="upper right", ncol=1, frameon=True)

    ax = axes[1]
    xq = q_s["period"].astype(str)
    ax.plot(xq, q_s["z_hpi_yoy"], lw=2.2, color="#0B4F6C", label="HPI YoY (quarterly)")
    ax.plot(xq, q_s["z_air_yoy"], lw=2.0, color="#E76F51", label="Air passenger YoY")
    try:
        x_list = list(xq)
        p0 = x_list.index("2020Q1")
        p1 = x_list.index("2021Q4")
        ax.axvspan(p0, p1, color="#F4A261", alpha=0.15, label="Pandemic period")
    except ValueError:
        pass
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ticks = np.linspace(0, len(xq) - 1, 8, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([x_list[i] for i in ticks], rotation=25, ha="right")
    ax.set_title("National quarterly averages (standardized)")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Z-score")
    ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Mobility and housing-price dynamics in the national panels", y=1.02, fontsize=12, fontweight="bold")
    fig.savefig(FIG_DIR / "fig_national_trends.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_event_study() -> None:
    es = pd.read_csv(RESULTS_DIR / "event_study_pretrend_coefficients.csv")
    meta = {x["model"]: x for x in json.loads((RESULTS_DIR / "event_study_pretrend_summary.json").read_text())}

    model_specs = [
        ("regional_event_study_open_persistent", "Persistent route openings", "#1D3557"),
        ("regional_event_study_close_persistent", "Persistent route closures", "#9C2F2F"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True, constrained_layout=True)
    for ax, (model, title, color) in zip(axes, model_specs):
        d = es[es["model"] == model].sort_values("event_time").copy()
        d["lo"] = d["coef"] - 1.96 * d["std_err"]
        d["hi"] = d["coef"] + 1.96 * d["std_err"]
        ax.axhline(0, color="black", lw=0.9, alpha=0.7)
        ax.axvline(-1, color="gray", lw=1.0, ls="--", alpha=0.7)
        ax.fill_between(d["event_time"], d["lo"], d["hi"], color=color, alpha=0.18)
        ax.plot(d["event_time"], d["coef"], color=color, lw=2)
        ax.scatter(d["event_time"], d["coef"], color=color, s=22, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Event time (quarters)")
        info = meta.get(model, {})
        pt = info.get("pretrend_joint_test", {}) or {}
        ax.text(
            0.02,
            0.04,
            f"Lead test p = {fmt_p(pt.get('p_value'))}\nTreated regions = {fmt_int(info.get('treated_regions'))}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
        )
        ax.set_xticks(np.arange(-8, 9, 2))
    axes[0].set_ylabel("Effect on regional RHPI YoY growth (pp)")
    fig.suptitle("Regional event-study estimates for persistent route shocks", y=1.03, fontsize=12, fontweight="bold")
    fig.savefig(FIG_DIR / "fig_event_study.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_route_shock_distribution() -> None:
    q = pd.read_parquet(PROC_DIR / "panel_quarterly_airport_shocks.parquet")
    qh = pd.read_parquet(PROC_DIR / "panel_quarterly_harmonized.parquet")
    if "period_str" not in q.columns and "period" in q.columns:
        q["period_str"] = q["period"].astype(str)
    if "period_str" not in qh.columns:
        qh["period_str"] = qh["year"].astype(int).astype(str) + "Q" + qh["quarter"].astype(int).astype(str)
    q = q.merge(qh[["geo", "period_str", "hpi_yoy_harmonized"]].drop_duplicates(["geo", "period_str"]), on=["geo", "period_str"], how="left")
    cols = ["open_persist_rate_norm_q", "close_persist_rate_norm_q"]
    d = q[cols].replace([np.inf, -np.inf], np.nan).copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        hi = d[c].quantile(0.99)
        d[c] = d[c].clip(lower=0, upper=hi)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    specs = [
        ("open_persist_rate_norm_q", "Persistent opening intensity", "#2A9D8F"),
        ("close_persist_rate_norm_q", "Persistent closure intensity", "#E76F51"),
    ]
    for ax, (col, title, color) in zip(axes, specs):
        x = d[col].dropna()
        ax.hist(x, bins=35, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("Normalized rate (winsorized at 99th pct.)")
        ax.set_ylabel("Country-quarter count")
        ax.axvline(x.median(), color="#1F1F1F", lw=1.2, ls="--", alpha=0.8, label=f"Median = {x.median():.3f}")
        ax.legend(frameon=True, loc="upper right")
    fig.suptitle("Distribution of national quarterly persistent route-shock intensities", y=1.02, fontsize=12, fontweight="bold")
    fig.savefig(FIG_DIR / "fig_route_shock_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_data_architecture() -> None:
    meta = {
        "panel": load_json(META_DIR / "panel_build_summary.json"),
        "shiftshare": load_json(META_DIR / "shiftshare_iv_summary.json"),
        "flight": load_json(META_DIR / "flight_shock_build_summary.json"),
        "cross": load_json(META_DIR / "airport_nuts2_crosswalk_summary.json"),
        "rega": load_json(META_DIR / "regional_panel_summary.json"),
        "regq": load_json(META_DIR / "regional_route_quarterly_panel_summary.json"),
    }
    labels = [
        "National annual panel",
        "National quarterly panel",
        "OD IV panel",
        "Route-month panel",
        "Country-quarter shocks",
        "NUTS2 annual panel",
        "NUTS2 quarterly panel",
    ]
    vals = [
        meta["panel"]["annual_rows"],
        meta["panel"]["quarterly_rows"],
        meta["shiftshare"]["iv_rows"],
        meta["flight"]["route_month_rows"],
        meta["flight"]["country_quarter_rows"],
        meta["rega"]["rows"],
        meta["regq"]["rows"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    ax = axes[0]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=["#264653", "#2A9D8F", "#457B9D", "#E9C46A", "#F4A261", "#E76F51", "#8D99AE"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("Rows (log scale)")
    ax.set_title("Dataset scale by pipeline stage")
    for yi, v in zip(y, vals):
        ax.text(v * 1.05, yi, fmt_int(v), va="center", fontsize=8.5)

    ax = axes[1]
    total = meta["cross"]["airports_total"]
    matched = meta["cross"]["nuts2_matched"]
    unmatched = total - matched
    ax.bar(["Matched", "Unmatched"], [matched, unmatched], color=["#2A9D8F", "#C44536"])
    ax.set_title("Airport-to-NUTS2 crosswalk match quality")
    ax.set_ylabel("Airports")
    ax.text(0, matched + 3, f"{matched}/{total}\n({100*matched/total:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.text(1, unmatched + 1, f"{unmatched}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(matched, unmatched) * 1.18)

    fig.suptitle("Research pipeline scale and spatial linkage coverage", y=1.03, fontsize=12, fontweight="bold")
    fig.savefig(FIG_DIR / "fig_data_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_country_heatmaps() -> None:
    geoms = load_country_polygons()
    if not geoms:
        return

    annual = pd.read_parquet(PROC_DIR / "panel_annual_harmonized.parquet").replace([np.inf, -np.inf], np.nan)
    annual["hpi_growth"] = pd.to_numeric(annual.get("hpi_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(annual.get("hpi_growth"), errors="coerce")
    )
    annual["L1_net_migration_rate"] = pd.to_numeric(annual.get("L1_net_migration_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(annual.get("L1_net_migration_rate"), errors="coerce")
    )
    if annual["L1_net_migration_rate"].isna().all() and "net_migration_rate_harmonized" in annual.columns:
        annual = annual.sort_values(["geo", "year"]).reset_index(drop=True)
        annual["L1_net_migration_rate"] = annual.groupby("geo", sort=False)["net_migration_rate_harmonized"].shift(1)
    annual = annual[(annual["year"] >= 2014) & (annual["year"] <= 2024)].copy()
    annual_means = annual.groupby("geo").agg(
        hpi_growth=("hpi_growth", "mean"),
        L1_net_migration_rate=("L1_net_migration_rate", "mean"),
    )

    q = pd.read_parquet(PROC_DIR / "panel_quarterly_harmonized.parquet").replace([np.inf, -np.inf], np.nan)
    if "period_str" not in q.columns:
        q["period_str"] = q["year"].astype(int).astype(str) + "Q" + q["quarter"].astype(int).astype(str)
    q = q[q["period_str"].astype(str).between("2006Q1", "2025Q3")]
    q_means = q.groupby("geo").agg(L1_air_yoy=("L1_air_yoy", "mean"))

    coef_base = pd.read_csv(RESULTS_DIR / "model_coefficients.csv")
    coef_adv = pd.read_csv(RESULTS_DIR / "advanced_model_coefficients.csv")
    beta_m = coef_lookup(coef_base, "annual_fe_full_controls", "L1_net_migration_rate")
    beta_a = coef_lookup(coef_adv, "quarterly_fe_airplus_routecounts", "L1_air_yoy")
    beta_m_val = float(beta_m["coef"]) if beta_m else 0.0
    beta_a_val = float(beta_a["coef"]) if beta_a else 0.0

    country = annual_means.join(q_means, how="outer")
    country["migr_implied_contrib_pp"] = beta_m_val * country["L1_net_migration_rate"]
    country["air_implied_contrib_pp_q"] = beta_a_val * country["L1_air_yoy"]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0), constrained_layout=True)
    _draw_country_choropleth(
        axes[0, 0],
        geoms,
        country["hpi_growth"],
        "Mean annual HPI growth (2014-2024, %)",
        cmap="YlOrRd",
        diverging=False,
    )
    _draw_country_choropleth(
        axes[0, 1],
        geoms,
        country["L1_net_migration_rate"],
        "Mean lagged net migration rate (per 1,000)",
        cmap="PuBuGn",
        diverging=True,
    )
    _draw_country_choropleth(
        axes[1, 0],
        geoms,
        country["migr_implied_contrib_pp"],
        "Model-implied migration contribution to HPI growth (pp)",
        cmap="RdBu_r",
        diverging=True,
    )
    _draw_country_choropleth(
        axes[1, 1],
        geoms,
        country["air_implied_contrib_pp_q"],
        "Model-implied air-mobility contribution (quarterly pp)",
        cmap="RdBu_r",
        diverging=True,
    )
    fig.suptitle(
        "Where mobility and housing-price pressures are strongest in Europe",
        y=1.02,
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig_country_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_airport_shock_map() -> None:
    geoms = load_country_polygons()
    cw_path = META_DIR / "airport_nuts2_crosswalk.csv"
    air_path = PROC_DIR / "airport_monthly_route_shocks.parquet"
    if not cw_path.exists() or not air_path.exists():
        return

    cw = pd.read_csv(cw_path)
    cw = cw.dropna(subset=["latitude_deg", "longitude_deg"]).copy()
    cw["rep_airp"] = cw["rep_airp"].astype(str)

    am = pd.read_parquet(air_path).replace([np.inf, -np.inf], np.nan)
    am["month"] = pd.PeriodIndex(am["month"].astype(str), freq="M")
    am = am[am["month"] >= pd.Period("2015-01", freq="M")].copy()
    am["net_persist_rate"] = am["open_persist_rate_norm"].fillna(0) - am["close_persist_rate_norm"].fillna(0)

    agg = (
        am.groupby("rep_airp", as_index=False)
        .agg(
            country=("country", "first"),
            avg_active_routes=("active_routes", "mean"),
            total_passengers=("passengers_total", "sum"),
            open_persist_total=("route_open_persist_m", "sum"),
            close_persist_total=("route_close_persist_m", "sum"),
            net_persist_rate=("net_persist_rate", "mean"),
        )
    )
    agg["net_persist_events"] = agg["open_persist_total"] - agg["close_persist_total"]
    agg = agg.merge(
        cw[["rep_airp", "airport_name", "municipality", "latitude_deg", "longitude_deg"]],
        on="rep_airp",
        how="left",
    )
    agg = agg.dropna(subset=["latitude_deg", "longitude_deg"]).copy()
    agg = agg[agg["avg_active_routes"].fillna(0) > 0].copy()
    if agg.empty:
        return

    size_base = np.sqrt(agg["total_passengers"].clip(lower=0)) / 75.0
    size_base = size_base.clip(lower=8, upper=180)
    agg["marker_size"] = size_base

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), constrained_layout=True)

    ax = axes[0]
    _draw_country_borders(ax, geoms)
    v = agg["net_persist_rate"].fillna(0)
    vmax = float(np.nanmax(np.abs(v))) if len(v) else 1.0
    if vmax == 0:
        vmax = 1.0
    sc = ax.scatter(
        agg["longitude_deg"],
        agg["latitude_deg"],
        c=agg["net_persist_rate"],
        s=agg["marker_size"],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Mean net persistent route-shock rate", fontsize=8)
    ax.set_title("Airport-level route-shock intensity (2015-2025)")

    # label a few major airports by passenger volume / shock salience
    lbl = agg.assign(score=agg["total_passengers"].rank(pct=True) + agg["net_persist_events"].abs().rank(pct=True))
    lbl = lbl.sort_values("score", ascending=False).head(14)
    for r in lbl.itertuples(index=False):
        code = str(r.rep_airp).split("_", 1)[-1]
        ax.text(float(r.longitude_deg) + 0.35, float(r.latitude_deg) + 0.15, code, fontsize=7, color="#222222", zorder=4)

    ax = axes[1]
    _draw_country_borders(ax, geoms, facecolor="#FBFBFB", edgecolor="#D8D8D8")
    country_agg = (
        agg.groupby("country", as_index=False)
        .agg(
            airports=("rep_airp", "nunique"),
            total_passengers=("total_passengers", "sum"),
            net_persist_events=("net_persist_events", "sum"),
            mean_net_persist_rate=("net_persist_rate", "mean"),
        )
        .sort_values("total_passengers", ascending=False)
        .head(12)
    )
    for i, r in enumerate(country_agg.itertuples(index=False), start=1):
        ax.text(
            -23,
            70 - i * 2.7,
            f"{i}. {r.country}: airports={int(r.airports)}, net events={int(round(r.net_persist_events))}, mean rate={r.mean_net_persist_rate:.3f}",
            fontsize=8,
            ha="left",
            va="center",
            family="monospace",
        )
    ax.set_title("Top countries by airport-route activity (summary)")

    fig.suptitle(
        "Which airports and countries drive the route-shock treatment variation",
        y=1.03,
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(FIG_DIR / "fig_airport_shock_map.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_key_coefficients() -> None:
    base = pd.read_csv(RESULTS_DIR / "model_coefficients.csv")
    adv = pd.read_csv(RESULTS_DIR / "advanced_model_coefficients.csv")

    specs = [
        ("Annual FE: migration only", base, "annual_fe_migration", "L1_net_migration_rate", "Migration"),
        ("Annual FE: + flights", base, "annual_fe_migration_flights", "L1_net_migration_rate", "Migration"),
        ("Annual FE: + controls", base, "annual_fe_full_controls", "L1_net_migration_rate", "Migration"),
        ("Country IV FE: composite", adv, "country_iv_fe", "L1_net_migration_rate", "Migration"),
        ("Country IV FE: asylum", adv, "country_iv_fe_asylum", "L1_net_migration_rate", "Migration"),
        ("Quarterly FE: air YoY (1 lag)", base, "quarterly_fe_air_yoy", "L1_air_yoy", "Air"),
        ("Quarterly FE: air YoY (2 lags)", base, "quarterly_fe_air_yoy_lags", "L1_air_yoy", "Air"),
        ("Quarterly FE + route counts", adv, "quarterly_fe_airplus_routecounts", "L1_air_yoy", "Air"),
        ("Quarterly FE + route rates", adv, "quarterly_fe_airplus_route_rates", "L1_air_yoy", "Air"),
    ]
    rows = []
    for label, df, model, term, group in specs:
        x = df[(df["model"] == model) & (df["term"] == term)]
        if x.empty:
            continue
        r = x.iloc[0]
        rows.append(
            {
                "label": label,
                "coef": r["coef"],
                "lo": r["coef"] - 1.96 * r["std_err"],
                "hi": r["coef"] + 1.96 * r["std_err"],
                "p": r["p_value"],
                "group": group,
            }
        )
    d = pd.DataFrame(rows)
    d["y"] = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    colors = d["group"].map({"Migration": "#B22222", "Air": "#1D3557"}).tolist()
    ax.hlines(d["y"], d["lo"], d["hi"], color=colors, lw=2)
    ax.scatter(d["coef"], d["y"], color=colors, s=40, zorder=3)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(d["y"])
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Coefficient estimate (95% CI)")
    ax.set_title("Key reduced-form and IV estimates across core specifications")
    # group labels
    for grp, color in [("Migration", "#B22222"), ("Air", "#1D3557")]:
        ax.scatter([], [], color=color, label=grp)
    ax.legend(frameon=True, loc="lower right")
    fig.savefig(FIG_DIR / "fig_key_coefficients.pdf", bbox_inches="tight")
    plt.close(fig)


def build_figures() -> None:
    plot_national_trends()
    plot_data_architecture()
    plot_country_heatmaps()
    plot_airport_shock_map()
    plot_route_shock_distribution()
    plot_event_study()
    plot_key_coefficients()


def build_main_tex() -> str:
    return r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{setspace}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{longtable}
\usepackage{tabularx}
\usepackage{array}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{float}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage[nameinlink,capitalize]{cleveref}
\usepackage{caption}
\usepackage{pdflscape}
\usepackage{siunitx}
\usepackage{natbib}
\usepackage{enumitem}
\usepackage{authblk}

\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{0pt}
\onehalfspacing
\hypersetup{
  colorlinks=true,
  linkcolor=blue!60!black,
  citecolor=blue!60!black,
  urlcolor=blue!60!black,
  pdftitle={Air Mobility, Migration, and House Prices in Europe},
  pdfauthor={Jose Juan de Leon and Francesca Romana Medda}
}

\title{\textbf{Do Migration and Air Mobility Raise House Prices in Europe?}\\[0.4em]
\large A Direct Empirical Answer from National Panels, Shift-Share IVs, and Route-Network Shocks}
\author[1,2]{Jose Juan de Leon\thanks{Primary and corresponding author. Email: \href{mailto:jose.guillamon.19@ucl.ac.uk}{jose.guillamon.19@ucl.ac.uk}. Phone: +44 07572 526024.}}
\author[3]{Francesca Romana Medda\thanks{Email: \href{mailto:f.medda@ucl.ac.uk}{f.medda@ucl.ac.uk}. Phone: +44 7805 828353.}}
\affil[1]{Dr. Quantitative Finance, University College London (UCL), UK}
\affil[2]{Quantitative Researcher, Equiti Group Capital, London, UK}
\affil[3]{Professor of Applied Economics and Finance; Director and Founder of the Institute of Finance and Technology, University College London (UCL), UK}
\date{}

\begin{document}
\maketitle

\begin{abstract}
A question that households, voters, and policymakers repeatedly ask is whether migration and cross-border mobility are making housing more expensive. This paper answers that question for Europe using a reproducible public-data pipeline that combines Eurostat house-price indices, migration series, and air passenger flows with origin-destination migration data, airport-partner route traffic, and regional OECD real house price indices. The design proceeds from baseline two-way fixed-effects (FE) national panels to a country-year shift-share instrumental variables (IV) strategy and regional route-shock event-study models.

The short answer is nuanced. In national FE panels, lagged net migration rates are positively associated with next-year house-price growth, and in quarterly national panels lagged air-passenger growth is positively associated with house-price growth. But once we move to country-year IV specifications, migration effects become imprecise and unstable across instrument sets, with diagnostics indicating weak-instrument and exclusion concerns in the overidentified specification. We further construct monthly airport-route opening/closure shocks and map airports to NUTS2 regions to estimate regional event studies. First-pass pre-trend diagnostics for persistent route opening and closure events do not reject joint lead equality, but event-time estimates remain imprecise in the corrected regional sample.

The paper therefore gives a more definitive answer than most public debate: mobility matters for housing dynamics, but the current public-data evidence does not support a credible one-factor claim that migration alone is driving European house-price inflation. Methodologically, the paper also provides a transparent framework for strengthening causal identification through exogenous route-shock tagging and improved refugee/asylum-based migration instruments.
\end{abstract}

\section{Introduction}
\label{sec:intro}

A central policy question across Europe is simple to state and hard to answer credibly: \emph{Are migration and increased mobility pushing up house prices?} The question matters to households facing affordability pressure, to local governments setting housing and infrastructure policy, and to national debates in which migration is often blamed for housing-market stress. Yet the underlying mechanisms are multiple, the relevant time scales differ, and causal identification is difficult with standard public data.

Housing markets in Europe are exposed to multiple mobility channels. Permanent population inflows can alter local housing demand, tenure choice, and expectations, while temporary mobility and connectivity (including air transport) can change accessibility, tourism intensity, and investment demand. These channels are conceptually distinct but empirically intertwined, and most public-data studies treat them separately.

This paper builds an integrated empirical framework linking migration, air mobility, and house-price dynamics using harmonized European data and a layered identification strategy. The goal is to provide a defensible answer to the public-facing housing question above while also documenting what can and cannot yet be claimed causally. The paper is written in the style of a modern empirical economics article: a question-first introduction, transparent identification discussion, headline estimates, and explicit limits to causal interpretation.

The paper makes three contributions. First, it constructs a reproducible multi-frequency dataset that links Eurostat national housing and mobility indicators, origin-destination migration flows, and monthly airport-partner route traffic. Second, it extends the national analysis with a country-year shift-share IV design and a route-shock event-study architecture designed to move from ``is it correlated?'' toward ``is it causal?'' Third, it upgrades the mobility treatment assignment to the regional level through an airport-to-NUTS2 spatial crosswalk and documents the implications of source-specific data-cleaning choices (notably OECD RHPI vintages).

\paragraph{Preview of findings.}
Three findings organize the paper. First, reduced-form national FE estimates show that migration and air mobility are both informative predictors of house-price growth, with the strongest and most stable signal in quarterly air-mobility regressions. Second, migration IV estimates are too weak and unstable to support strong causal claims in the current sample. Third, the route-shock event-study design is promising because timing assumptions can be tested directly, but it still requires more exogenous event tagging to become publication-grade causal evidence.

\section{Related Literature}
\label{sec:lit}

The migration--housing literature commonly uses two-way FE panels, spatial models, or shift-share/Bartik-style IVs based on pre-existing settlement patterns and origin-country shocks. \citet{saiz2008immigration} emphasizes the role of housing supply elasticity in shaping price responses to immigration. Regional analyses such as \citet{degenfischer2017} and \citet{helfer2023migration} show that migration effects can be heterogeneous across local housing markets.

Work on air connectivity and urban outcomes often relies on quasi-experiments in route supply or airport access. \citet{blonigencristea2015} identify effects of air service on urban growth, while recent housing-related work highlights how mobility and tourism demand can transmit into local real estate prices (\citealp{tomal2025}). These approaches motivate the route-shock event-study component of our design.

\section{Data and Measurement}
\label{sec:data}

\subsection{Data Sources}

The empirical pipeline combines Eurostat, OECD, and World Bank sources. National panels use Eurostat house price indices (annual and quarterly), net migration rates, migration flow aggregates, air passenger volumes, and macroeconomic controls. Advanced components use Eurostat origin-destination immigration data (\texttt{MIGR\_IMM5PRV}), first-time asylum applications by citizenship and destination (\texttt{MIGR\_ASYAPPCTZA}), and monthly airport-partner passenger traffic (\texttt{AVIA\_PAOAC}). Regional outcomes come from the OECD real house price index (RHPI) dataflow.

\input{tables/tab_coverage.tex}

Table~\ref{tab:coverage} summarizes the empirical architecture. The key design choice is to study two mobility channels jointly but at different frequencies: migration as a slower-moving population inflow measure and air traffic as a high-frequency mobility/connectivity measure. This is one way the paper differs from most existing studies, which typically isolate one channel.

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/fig_country_heatmaps.pdf}
    \caption{Europe-wide heat maps: housing outcomes, mobility exposures, and model-implied contributions (coefficient $\times$ average exposure)}
    \label{fig:country_heatmaps}
\end{figure}

Figure~\ref{fig:country_heatmaps} provides the visual backbone of the paper: it shows where house-price growth is strongest, where migration exposure is high, and how large the model-implied migration and air-mobility contributions are. The visual message anticipates the regression results: measured mobility contributions are generally present but modest relative to total observed house-price growth.

\subsection{Key Variable Construction}

The main national outcomes are annual house-price growth and quarterly year-over-year (YoY) house-price growth. Air mobility is measured as log growth (annual) or YoY growth (quarterly) in passenger volumes. Migration is measured primarily as the crude rate of net migration plus statistical adjustment (per 1,000 inhabitants).

For the route-shock design, we construct monthly airport-partner route activity indicators and define route openings/closures based on transitions in active status. We then classify \emph{persistent} openings and closures using forward-looking activity windows to distinguish durable network changes from short-lived fluctuations, and aggregate these shocks to country-quarter and region-quarter intensity measures.

For the regional RHPI data, a critical preprocessing step is to avoid duplicate region-period observations introduced by the OECD \texttt{VINTAGE} dimension (e.g., \texttt{NEW}, \texttt{EXISTING}, and total stock). We retain total-stock observations (\texttt{VINTAGE=\_T}) and non-seasonally adjusted data (\texttt{ADJUSTMENT=N}) in the regional panels used for FE and event-study analysis.

\input{tables/tab_desc.tex}

Table~\ref{tab:desc} shows that house-price growth is volatile relative to the average magnitude of annual migration-rate changes, which is one reason why economically meaningful but statistically modest coefficients are plausible in this setting.

\section{Empirical Strategy}
\label{sec:methods}

\subsection{National Baseline FE Models}

We begin with country and time fixed-effects models:
\begin{equation}
y_{it} = \beta m_{i,t-1} + \gamma a_{i,t-1} + X'_{i,t-1}\delta + \mu_i + \tau_t + \varepsilon_{it},
\end{equation}
where $y_{it}$ is house-price growth, $m_{i,t-1}$ is lagged migration exposure, $a_{i,t-1}$ is lagged air mobility, $X_{i,t-1}$ includes macro controls, $\mu_i$ are country fixed effects, and $\tau_t$ are year (or quarter) fixed effects.

These models provide reduced-form benchmarks but do not address endogeneity in migration or air mobility.

\subsection{Country-Year Shift-Share IV Design}

To address migration endogeneity, we estimate 2SLS models with country and year FE using shift-share instruments built from (i) pre-period destination-specific origin shares and (ii) origin-level shocks. Two instrument variants are used:
\begin{enumerate}[leftmargin=1.3em]
\item a composite specification combining a World Bank push-shock index and a leave-one-out origin-supply component, and
\item an asylum-specific leave-one-out shift-share exposure based on first-time asylum applications by origin.
\end{enumerate}

The overidentified composite specification allows over-identification diagnostics; the asylum specification provides a cleaner exclusion narrative but can be weaker in finite samples.

\subsection{Route-Shock Event-Study Design}

We construct monthly airport-route opening and closure events, classify persistent events, and aggregate them to regional-quarter shock intensities through an airport-to-NUTS2 crosswalk. Event-study models use the first qualifying event in each region, a symmetric eight-quarter window, and omit event time $-1$:
\begin{equation}
y_{rt} = \sum_{k \in \{-8,\ldots,8\}\setminus\{-1\}} \theta_k \mathbf{1}\{t-T_r = k\} + \alpha_r + \lambda_t + u_{rt}.
\end{equation}
We report individual event-time coefficients and a joint Wald test of pre-trend leads ($k=-8,\ldots,-2$). The current implementation uses a transparent TWFE event-study as a baseline, while future versions should incorporate modern staggered-adoption estimators emphasized in recent top-journal econometrics work (e.g., \citealp{borusyak2024revisiting}).

\section{Results}
\label{sec:results}

\subsection{National FE Benchmarks}

Table~\ref{tab:baseline_annual} reports the national annual FE benchmark models. The migration coefficient is consistently positive, with point estimates around 0.10 percentage points of next-year house-price growth for one additional net migrant per 1,000 residents. This is a small average effect in absolute terms, but it is not trivial: a sustained increase of 5 net migrants per 1,000 is associated with roughly 0.5 percentage points higher annual house-price growth in the full-controls specification. The coefficient remains stable after adding annual flight growth and macro controls.

By contrast, annual air-passenger growth is not statistically significant in the annual national models. This suggests that the annual migration relationship is not simply proxying for broad annual mobility trends.

\input{tables/tab_baseline_annual.tex}

Figure~\ref{fig:key_coefs} places these annual migration estimates next to the quarterly air-mobility estimates and the IV estimates, making the paper's main empirical pattern immediately visible: reduced-form signals are present, but causal estimates are weaker and less precise.

The quarterly national results in Table~\ref{tab:quarterly_models} tell a complementary story. Lagged air-passenger YoY growth is positively associated with house-price YoY growth, and the coefficient remains positive and significant when route-shock controls are added. In magnitude terms, a 10 percentage-point increase in air-passenger YoY growth corresponds to roughly 0.4 percentage points higher house-price YoY growth in the route-shock FE models. This is again a modest average effect, but it is precisely estimated relative to the annual migration FE signal.

The route-shock variables themselves are not significant in country-quarter aggregates, which is consistent with substantial within-country heterogeneity and motivates the spatial mapping and regional event-study analysis below.

\input{tables/tab_quarterly.tex}

Figure~\ref{fig:national_trends} shows why this quarterly result is empirically relevant: air-mobility and house-price dynamics co-move strongly around major episodes, including the pandemic disruption and recovery. Figure~\ref{fig:route_dist} and Figure~\ref{fig:airport_shock_map} then show how route-shock variation is distributed across countries and airports, illustrating that the identifying variation is highly concentrated in a subset of hubs and national systems.

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig_national_trends.pdf}
    \caption{Average national mobility and house-price dynamics in annual and quarterly panels (standardized)}
    \label{fig:national_trends}
\end{figure}

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.90\textwidth]{figures/fig_key_coefficients.pdf}
    \caption{Key coefficient estimates across national FE and IV specifications}
    \label{fig:key_coefs}
\end{figure}

\subsection{Country-Year IV Estimates and Diagnostics}

Tables~\ref{tab:iv_results} and \ref{tab:iv_diag} provide the paper's main causal-discipline check. The country-year IV estimates are substantially less precise than the reduced-form FE estimates. In both the composite and asylum-specific IV specifications, the migration coefficient is statistically insignificant and its sign is unstable across instrument sets.

\input{tables/tab_iv_results.tex}

Diagnostics indicate that the composite IV specification has weak first-stage relevance by conventional thresholds and fails over-identification tests, consistent with potential exclusion violations in at least one component. The asylum IV is conceptually cleaner but also weak in this sample. This is exactly why the paper keeps the reduced-form estimates and the route-shock design in the same narrative: the evidence is informative, but the migration-only causal claim is not yet definitive.

\input{tables/tab_iv_diag.tex}

\subsection{Regional FE and Event-Study Evidence}

After correcting the OECD RHPI regional sample for duplicate vintages, the regional NUTS2 annual panel is much smaller than the preliminary version, and the regional FE estimates become more fragile. Table~\ref{tab:regional_fe} should therefore be read as exploratory evidence on regional heterogeneity rather than a headline result.

\input{tables/tab_regional_fe.tex}

The regional quarterly route-shock event-study design remains viable in the corrected sample and provides a stronger test of timing assumptions. Table~\ref{tab:event_pretrend} and Figure~\ref{fig:event_study} show that joint pre-trend tests do not reject for persistent route opening or closure events, although the opening-event lead coefficients are individually elevated in several pre-period quarters, indicating that remaining selection or dynamic anticipation cannot be ruled out.

\input{tables/tab_event_pretrend.tex}

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/fig_airport_shock_map.pdf}
    \caption{Where route-shock treatment variation comes from: airport-level intensity and country context}
    \label{fig:airport_shock_map}
\end{figure}

Figure~\ref{fig:airport_shock_map} is included deliberately as part of the identification argument, not only as a descriptive map. It shows where the treatment variation originates (specific airports and national airport systems), making it easier to evaluate whether route shocks are plausibly exogenous or instead driven by underlying local demand conditions.

Figure~\ref{fig:data_arch} complements this by showing the scale of each dataset layer in the pipeline and the coverage gain/loss as the analysis moves from national panels to spatially mapped regional designs.

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig_event_study.pdf}
    \caption{Regional event-study coefficients for persistent route openings and closures}
    \label{fig:event_study}
\end{figure}

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig_data_architecture.pdf}
    \caption{Pipeline scale and airport-to-region mapping coverage}
    \label{fig:data_arch}
\end{figure}

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig_route_shock_distribution.pdf}
    \caption{Distribution of persistent route-shock intensity measures (country-quarter)}
    \label{fig:route_dist}
\end{figure}

\section{Discussion, Limitations, and Next Steps}
\label{sec:discussion}

The current evidence supports three main conclusions that directly answer the motivating question.
\begin{enumerate}[leftmargin=1.3em]
\item National reduced-form FE estimates suggest that migration and air mobility are relevant correlates of housing-price dynamics, especially in the quarterly air-mobility specifications; average implied effect sizes are modest rather than explosive.
\item Country-year IV estimates currently lack sufficient precision and credibility for strong causal claims about migration's causal effect. The composite instrument shows both weak first-stage strength and over-identification rejection; the asylum-based instrument offers a cleaner exclusion story but remains weak.
\item The route-shock event-study architecture is promising because it leverages timing variation and permits explicit pre-trend diagnostics. However, stronger exogeneity requires tagging route events to plausibly external shocks (regulatory, slot-allocation, airline-network, or disruption events), and future versions should pair this with newer event-study estimators rather than relying only on TWFE.
\end{enumerate}

The next research priority is to attach exogenous labels to persistent route events and to refine the migration IV around refugee/asylum shocks with stronger institutional exclusion arguments. A second priority is improving airport exposure measurement from administrative NUTS2 assignment to catchment-based, travel-time-weighted exposure. A third is to reorganize the empirical design around a ``channel decomposition'' contribution: migration as the slower population channel, air connectivity as the faster accessibility channel, and housing prices as the common outcome.

\section{Conclusion}
\label{sec:conclusion}

This paper provides an evidence-based answer to a widely asked housing question: migration and mobility are associated with higher house-price growth in several reduced-form specifications, but the estimated average impacts are modest and the current public-data causal evidence is not strong enough to justify simple one-line claims about migration as the dominant driver of housing inflation. The national FE evidence identifies robust correlations; the IV and event-study modules show exactly where the causal case is weak, and where the design can be strengthened. That is a more useful answer for policy and research than either denial or overstatement.

\clearpage
\appendix
\section*{Appendix: Reproducibility Notes}

The manuscript and assets were generated from repository outputs with scripted table/figure production. The underlying pipeline includes Eurostat pulls, processed national and regional panels, IV construction, route-shock construction, and FE/event-study estimation. See the repository scripts and documentation for the full run order.

\bibliographystyle{apalike}
\bibliography{references}

\end{document}
""".lstrip()


def build_references_bib() -> str:
    return r"""
@misc{eurostat_hpi,
  author = {{Eurostat}},
  title = {House price index (HPI) datasets and documentation},
  year = {2026},
  howpublished = {\url{https://ec.europa.eu/eurostat/databrowser/view/PRC_HPI_Q/default/table?lang=en}},
  note = {Accessed February 26, 2026}
}

@misc{eurostat_avia,
  author = {{Eurostat}},
  title = {Air passenger transport statistics and AVIA tables},
  year = {2026},
  howpublished = {\url{https://ec.europa.eu/eurostat/databrowser/view/AVIA_PAOC/default/table?lang=en}},
  note = {Accessed February 26, 2026}
}

@misc{eurostat_migration,
  author = {{Eurostat}},
  title = {Migration and asylum in Europe (interactive publication)},
  year = {2026},
  howpublished = {\url{https://ec.europa.eu/eurostat/web/interactive-publications/migration-2023}},
  note = {Accessed February 26, 2026}
}

@misc{oecd_rhpi,
  author = {{OECD}},
  title = {Real House Price Index (RHPI) dataflow and API endpoint},
  year = {2026},
  howpublished = {\url{https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_RHPI@DF_RHPI_ALL,1.0/.?startPeriod=1990&format=csvfile}},
  note = {Accessed February 26, 2026}
}

@misc{worldbank_api,
  author = {{World Bank}},
  title = {Indicators API Documentation},
  year = {2026},
  howpublished = {\url{https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation}},
  note = {Accessed February 26, 2026}
}

@misc{ourairports,
  author = {{OurAirports}},
  title = {OurAirports airport data catalog},
  year = {2026},
  howpublished = {\url{https://ourairports.com/data/airports.csv}},
  note = {Accessed February 26, 2026}
}

@misc{gisco_nuts,
  author = {{Eurostat GISCO}},
  title = {NUTS GeoJSON distribution service},
  year = {2026},
  howpublished = {\url{https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/}},
  note = {Accessed February 26, 2026}
}

@techreport{saiz2008immigration,
  author = {Saiz, Albert},
  title = {Immigration and Housing Rents in American Cities},
  institution = {National Bureau of Economic Research},
  year = {2008},
  number = {14188},
  url = {https://www.nber.org/papers/w14188}
}

@article{blonigencristea2015,
  author = {Blonigen, Bruce A. and Cristea, Anca D.},
  title = {Air Service and Urban Growth: Evidence from a Quasi-Natural Policy Experiment},
  journal = {Journal of Urban Economics},
  year = {2015},
  volume = {86},
  pages = {128--146}
}

@article{degenfischer2017,
  author = {Degen, Kathrin and Fischer, Andreas M.},
  title = {Immigration and Swiss House Prices},
  journal = {Swiss Journal of Economics and Statistics},
  year = {2017},
  volume = {153},
  number = {1},
  pages = {15--36}
}

@article{helfer2023migration,
  author = {Helfer, Lukas and Hlavac, Marek and Schmidpeter, Bernhard},
  title = {Migration and House Prices: A Regional Perspective},
  journal = {Regional Science and Urban Economics},
  year = {2023},
  volume = {101},
  pages = {103912}
}

@article{borusyak2024revisiting,
  author = {Borusyak, Kirill and Jaravel, Xavier and Spiess, Jann},
  title = {Revisiting Event Study Designs: Robust and Efficient Estimation},
  journal = {Review of Economic Studies},
  year = {2024},
  volume = {91},
  number = {6},
  pages = {3253--3285}
}

@misc{zhupryce2025,
  author = {Zhu, Bei and Pryce, Gwilym},
  title = {Migration and House Prices in England and Wales: Spatial Panel IV Evidence},
  year = {2025},
  howpublished = {Accepted manuscript record, White Rose Research Online},
  url = {https://eprints.whiterose.ac.uk/id/eprint/221511/}
}

@article{tomal2025,
  author = {Tomal, Mateusz},
  title = {Tourism and Housing Prices: Panel Evidence from European Regions},
  journal = {Journal of Real Estate Finance and Economics},
  year = {2025},
  note = {Online first / 2025 record}
}
""".lstrip()


def build_readme_upload() -> str:
    return """Overleaf Upload Package

Contents:
- `main.tex` (paper manuscript)
- `references.bib` (bibliography)
- `figures/` (publication-style figures)
- `tables/` (LaTeX tables generated from local results)

Notes:
- This package is generated from the current repository outputs (including corrected OECD RHPI regional filtering for VINTAGE=_T and ADJUSTMENT=N).
- Edit author/affiliation in `main.tex` before submission.
- Compile order on Overleaf: PDFLaTeX -> BibTeX -> PDFLaTeX -> PDFLaTeX (Overleaf usually handles this automatically).
"""


def generate_tables() -> None:
    write_text(TAB_DIR / "tab_coverage.tex", build_table_coverage())
    write_text(TAB_DIR / "tab_desc.tex", build_table_descriptive())
    write_text(TAB_DIR / "tab_baseline_annual.tex", build_table_baseline_annual())
    write_text(TAB_DIR / "tab_quarterly.tex", build_table_quarterly())
    write_text(TAB_DIR / "tab_iv_results.tex", build_table_iv_results())
    write_text(TAB_DIR / "tab_iv_diag.tex", build_table_iv_diagnostics())
    regional, event = build_table_regional_and_event()
    write_text(TAB_DIR / "tab_regional_fe.tex", regional)
    write_text(TAB_DIR / "tab_event_pretrend.tex", event)


def build_package() -> None:
    if PAPER_DIR.exists():
        shutil.rmtree(PAPER_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    build_figures()
    generate_tables()
    write_text(PAPER_DIR / "main.tex", build_main_tex())
    write_text(PAPER_DIR / "references.bib", build_references_bib())
    write_text(PAPER_DIR / "README_UPLOAD.txt", build_readme_upload())


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=PAPER_DIR)


def main() -> None:
    build_package()
    build_zip()
    manifest = {
        "paper_dir": str(PAPER_DIR.relative_to(ROOT)),
        "zip_path": str(ZIP_PATH.relative_to(ROOT)),
        "figures": sorted(p.name for p in FIG_DIR.glob("*")),
        "tables": sorted(p.name for p in TAB_DIR.glob("*.tex")),
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
