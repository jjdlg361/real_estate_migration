#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PROC_DIR = ROOT / "data" / "processed"
META_DIR = ROOT / "data" / "metadata"
TABLE_DIR = ROOT / "paper_overleaf" / "tables"
AUDIT_DIR = RESULTS_DIR / "paper_table_csv"


def latex_escape(s: Any) -> str:
    if s is None:
        return ""
    x = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        x = x.replace(k, v)
    return x


def fmt_int(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)


def fmt_num(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:.{digits}f}"


def fmt_coef(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    v = float(x)
    if abs(v) < 0.001 and v != 0:
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return f"{v:.3f}"


def sig_stars(p: Any) -> str:
    if p is None:
        return ""
    try:
        pv = float(p)
    except Exception:
        return ""
    if np.isnan(pv):
        return ""
    if pv < 0.01:
        return "***"
    if pv < 0.05:
        return "**"
    if pv < 0.10:
        return "*"
    return ""


def fmt_se(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    v = float(x)
    if abs(v) < 0.001 and v != 0:
        return f"({v:.6f}".rstrip("0").rstrip(".") + ")"
    return f"({v:.3f})"


def write_long_csv(table_name: str, rows: list[dict[str, Any]]) -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_DIR / f"{table_name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def render_long_table(
    csv_path: Path,
    *,
    caption: str,
    label: str,
    tabular_spec: str,
    header_cols: list[tuple[str, str]],
    note_lines: list[str],
    resize_to_textwidth: bool = False,
    small: bool = True,
    set_tabcolsep: int | None = None,
) -> str:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty table source: {csv_path}")
    df["row_label"] = df["row_label"].fillna("")

    # Preserve row order exactly as written to CSV.
    row_meta = (
        df[["row_order", "row_label", "row_kind"]]
        .drop_duplicates()
        .sort_values(["row_order", "row_kind"], kind="stable")
        .reset_index(drop=True)
    )

    pivot = (
        df.pivot_table(
            index=["row_order", "row_label", "row_kind"],
            columns="col_key",
            values="display",
            aggfunc="first",
        )
        .reset_index()
        .sort_values("row_order", kind="stable")
    )

    col_keys = [k for k, _ in header_cols]
    lines: list[str] = []
    for _, r in pivot.iterrows():
        vals = ["" if pd.isna(r.get(k)) else str(r.get(k)) for k in col_keys]
        label_cell = "" if pd.isna(r["row_label"]) else str(r["row_label"])
        if str(r["row_kind"]) == "midrule":
            lines.append(r"\midrule")
            continue
        lines.append(latex_escape(label_cell) + " & " + " & ".join(vals) + r" \\")

    header = " & ".join(latex_escape(lbl) for _, lbl in header_cols)
    wrapper_open = ""
    wrapper_close = ""
    if resize_to_textwidth:
        wrapper_open = r"\resizebox{\textwidth}{!}{%" + "\n"
        wrapper_close = "\n}"

    parts = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{threeparttable}",
    ]
    if small:
        parts.append(r"\small")
    if set_tabcolsep is not None:
        parts.append(rf"\setlength{{\tabcolsep}}{{{set_tabcolsep}pt}}")
    if wrapper_open:
        parts.append(wrapper_open.rstrip("\n"))
    parts += [
        rf"\begin{{tabular}}{{{tabular_spec}}}",
        r"\toprule",
        " & " + header + r" \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if wrapper_close:
        parts.append(wrapper_close)
    parts += [r"\begin{tablenotes}[flushleft]", r"\footnotesize"]
    for line in note_lines:
        parts.append(rf"\item {line}")
    parts += [r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(parts)


def coef_lookup(df: pd.DataFrame, model: str, term: str) -> dict[str, float] | None:
    sub = df[(df["model"] == model) & (df["term"] == term)]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return {
        "coef": float(r["coef"]) if pd.notna(r["coef"]) else np.nan,
        "std_err": float(r["std_err"]) if pd.notna(r["std_err"]) else np.nan,
        "p_value": float(r["p_value"]) if pd.notna(r["p_value"]) else np.nan,
        "nobs": float(r["nobs"]) if pd.notna(r["nobs"]) else np.nan,
        "r2_within": float(r["r2_within"]) if "r2_within" in r.index and pd.notna(r["r2_within"]) else np.nan,
    }


def add_coef_rows(
    rows: list[dict[str, Any]],
    row_order: float,
    row_label: str,
    term: str,
    models: list[tuple[str, str]],
    source_df: pd.DataFrame,
) -> None:
    coef_entries = []
    se_entries = []
    for col_key, model in models:
        cell = coef_lookup(source_df, model, term)
        coef_entries.append((col_key, cell))
        se_entries.append((col_key, cell))
    for col_key, cell in coef_entries:
        rows.append(
            {
                "row_order": row_order,
                "row_kind": "coef",
                "row_label": row_label,
                "col_key": col_key,
                "display": "" if cell is None else (fmt_coef(cell["coef"]) + sig_stars(cell["p_value"])),
                "source_model": model if False else "",  # populated below in audit loop not needed in display
                "source_term": term,
                "coef": np.nan if cell is None else cell["coef"],
                "std_err": np.nan if cell is None else cell["std_err"],
                "p_value": np.nan if cell is None else cell["p_value"],
            }
        )
    for col_key, cell in se_entries:
        rows.append(
            {
                "row_order": row_order + 0.001,
                "row_kind": "se",
                "row_label": "",
                "col_key": col_key,
                "display": "" if cell is None else fmt_se(cell["std_err"]),
                "source_model": "",
                "source_term": term,
                "coef": np.nan if cell is None else cell["coef"],
                "std_err": np.nan if cell is None else cell["std_err"],
                "p_value": np.nan if cell is None else cell["p_value"],
            }
        )


def add_stat_row(rows: list[dict[str, Any]], row_order: float, row_label: str, values: dict[str, str]) -> None:
    for col_key, display in values.items():
        rows.append(
            {
                "row_order": row_order,
                "row_kind": "stat",
                "row_label": row_label,
                "col_key": col_key,
                "display": display,
                "source_model": "",
                "source_term": "",
                "coef": np.nan,
                "std_err": np.nan,
                "p_value": np.nan,
            }
        )


def add_midrule(rows: list[dict[str, Any]], row_order: float, col_keys: list[str]) -> None:
    for col_key in col_keys:
        rows.append(
            {
                "row_order": row_order,
                "row_kind": "midrule",
                "row_label": "",
                "col_key": col_key,
                "display": "",
                "source_model": "",
                "source_term": "",
                "coef": np.nan,
                "std_err": np.nan,
                "p_value": np.nan,
            }
        )


def build_table_baseline_annual() -> tuple[Path, str]:
    coef = pd.read_csv(RESULTS_DIR / "model_coefficients.csv")
    models = [
        ("m1", "annual_fe_migration"),
        ("m2", "annual_fe_migration_flights"),
        ("m3", "annual_fe_full_controls"),
    ]
    headers = [("m1", "Migration only"), ("m2", "+ Flights"), ("m3", "+ Full controls")]
    rows: list[dict[str, Any]] = []
    add_coef_rows(rows, row_order=1, row_label="Lagged net migration rate", term="L1_net_migration_rate", models=models, source_df=coef)
    add_coef_rows(rows, row_order=2, row_label="Lagged air-passenger growth", term="L1_air_growth", models=models, source_df=coef)
    add_coef_rows(rows, row_order=3, row_label="Lagged GDP per-capita growth", term="L1_gdp_pc_growth", models=models, source_df=coef)
    add_coef_rows(rows, row_order=4, row_label="Lagged unemployment rate", term="L1_unemployment_rate", models=models, source_df=coef)
    add_coef_rows(rows, row_order=5, row_label="Lagged inflation", term="L1_inflation_hicp", models=models, source_df=coef)
    add_coef_rows(rows, row_order=6, row_label="Lagged long-term rate", term="L1_long_rate", models=models, source_df=coef)
    add_coef_rows(rows, row_order=7, row_label="Lagged population growth", term="L1_pop_growth", models=models, source_df=coef)
    add_midrule(rows, 90, [k for k, _ in headers])
    add_stat_row(rows, 91, "Country FE", {"m1": "Yes", "m2": "Yes", "m3": "Yes"})
    add_stat_row(rows, 92, "Year FE", {"m1": "Yes", "m2": "Yes", "m3": "Yes"})
    add_stat_row(rows, 93, "Clustered SE (country)", {"m1": "Yes", "m2": "Yes", "m3": "Yes"})
    add_stat_row(rows, 94, "Macro controls included", {"m1": "No", "m2": "No", "m3": "Yes"})
    add_stat_row(
        rows,
        95,
        "Observations",
        {k: fmt_int(coef[coef["model"] == m]["nobs"].dropna().iloc[0]) for k, m in models},
    )
    add_stat_row(
        rows,
        96,
        "R-squared (within)",
        {k: fmt_num(coef[coef["model"] == m]["r2_within"].dropna().iloc[0], 3) for k, m in models},
    )
    csv_path = write_long_csv("tab_baseline_annual", rows)
    tex = render_long_table(
        csv_path,
        caption="Baseline annual two-way fixed-effects estimates (national panel)",
        label="tab:baseline_annual",
        tabular_spec="p{5.4cm}ccc",
        header_cols=headers,
        note_lines=[
            r"Notes: Dependent variable is annual house-price growth. All regressors are lagged one period. Standard errors (in parentheses) are clustered at the country level. Significance stars are computed directly from source-model p-values in the audited CSV pipeline (* $p<0.10$, ** $p<0.05$, *** $p<0.01$); exact p-values are available in \texttt{results/model\_coefficients.csv} and the audited table CSV.",
        ],
    )
    return csv_path, tex


def build_table_quarterly() -> tuple[Path, str]:
    base = pd.read_csv(RESULTS_DIR / "model_coefficients.csv")
    adv = pd.read_csv(RESULTS_DIR / "advanced_model_coefficients.csv")
    both = pd.concat([base, adv], ignore_index=True)
    models = [
        ("m1", "quarterly_fe_air_yoy"),
        ("m2", "quarterly_fe_air_yoy_lags"),
        ("m3", "quarterly_fe_airplus_routecounts"),
        ("m4", "quarterly_fe_airplus_route_rates"),
    ]
    headers = [("m1", "Air YoY (1 lag)"), ("m2", "Air YoY (2 lags)"), ("m3", "+ Route counts"), ("m4", "+ Route rates")]
    rows: list[dict[str, Any]] = []
    add_coef_rows(rows, 1, "Lagged air-passenger YoY", "L1_air_yoy", models, both)
    add_coef_rows(rows, 2, "2nd lag air-passenger YoY", "L2_air_yoy", models, both)
    add_coef_rows(rows, 3, "Lagged net route openings (count)", "L1_net_openings_q", models, both)
    add_coef_rows(rows, 4, "Lagged persistent opening intensity", "L1_open_rate_norm_q", models, both)
    add_coef_rows(rows, 5, "Lagged persistent closure intensity", "L1_close_rate_norm_q", models, both)
    add_midrule(rows, 90, [k for k, _ in headers])
    add_stat_row(rows, 91, "Country FE", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 92, "Quarter FE", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 93, "Two-way clustered SE", {"m1": "No", "m2": "No", "m3": "Yes", "m4": "Yes"})
    add_stat_row(rows, 94, "Observations", {k: fmt_int(both[both["model"] == m]["nobs"].dropna().iloc[0]) for k, m in models})
    csv_path = write_long_csv("tab_quarterly", rows)
    tex = render_long_table(
        csv_path,
        caption="Quarterly national fixed-effects estimates: air mobility and route-shock extensions",
        label="tab:quarterly_models",
        tabular_spec="p{5.8cm}cccc",
        header_cols=headers,
        note_lines=[
            r"Notes: Dependent variable is quarterly house-price YoY growth. Baseline models use country-clustered standard errors; route-shock FE models use two-way clustering by country and quarter. Persistent route-event intensity variables come from monthly airport-partner route shocks aggregated to the country-quarter level. Significance stars are computed directly from source-model p-values in the audited CSV pipeline (* $p<0.10$, ** $p<0.05$, *** $p<0.01$); exact p-values are in the source result CSVs and the audited table CSV.",
        ],
        resize_to_textwidth=True,
    )
    return csv_path, tex


def build_table_traveler_quality() -> tuple[Path, str]:
    coef = pd.read_csv(RESULTS_DIR / "traveler_quality_coefficients.csv")
    models = [
        ("m1", "quarterly_tq_baseline_matched"),
        ("m2", "quarterly_tq_plus_airfare"),
        ("m3", "quarterly_tq_full"),
        ("m4", "quarterly_tq_full_interact"),
    ]
    headers = [("m1", "Matched baseline"), ("m2", "+ Airfare"), ("m3", "Full TQ"), ("m4", "Full TQ + int.")]
    rows: list[dict[str, Any]] = []
    for i, (term, label) in enumerate(
        [
            ("L1_air_yoy", "Lagged air-passenger YoY growth"),
            ("L1_airfare_yoy_q", "Lagged airfare inflation (HICP CP0733, YoY)"),
            ("L1_lic_neu_share_pas", "Lagged non-EU carrier passenger share"),
            ("L1_pax_per_move_total", "Lagged passengers per commercial movement"),
            ("L1_air_yoy:L1_airfare_yoy_q_c", "Lagged air growth × demeaned airfare inflation"),
            ("L1_air_yoy:L1_lic_neu_share_pas_c", "Lagged air growth × demeaned non-EU carrier share"),
            ("L1_open_rate_norm_q", "Lagged route opening rate"),
            ("L1_close_rate_norm_q", "Lagged route closure rate"),
        ],
        start=1,
    ):
        add_coef_rows(rows, i, label, term, models, coef)
    add_midrule(rows, 90, [k for k, _ in headers])
    add_stat_row(rows, 91, "Country FE", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 92, "Quarter FE", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 93, "Two-way clustered SE", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 94, "Observations", {k: fmt_int(coef[coef["model"] == m]["nobs"].dropna().iloc[0]) for k, m in models})
    csv_path = write_long_csv("tab_traveler_quality_quarterly", rows)
    tex = render_long_table(
        csv_path,
        caption="Traveler-quality proxies in quarterly country FE models",
        label="tab:traveler_quality_quarterly",
        tabular_spec="p{5.9cm}p{2.15cm}p{2.15cm}p{2.15cm}p{2.15cm}",
        header_cols=headers,
        note_lines=[
            r"Notes: Dependent variable is quarterly house-price YoY growth. All models include country and quarter fixed effects and two-way clustered standard errors (country and quarter). Airfare inflation is based on Eurostat HICP \texttt{CP0733} (air passenger transport). Carrier-mix proxies are aggregated from airport-level \texttt{AVIA\_TF\_APAL} data using airline licence categories (\texttt{LIC\_EU} vs \texttt{LIC\_NEU}). Significance stars are computed directly from source-model p-values in the audited CSV pipeline (* $p<0.10$, ** $p<0.05$, *** $p<0.01$).",
        ],
        set_tabcolsep=4,
    )
    return csv_path, tex


def build_table_who_arrives() -> tuple[Path, str]:
    coef = pd.read_csv(RESULTS_DIR / "migration_composition_coefficients.csv")
    models = [
        ("m1", "annual_fe_netmig_controls_matchedsample"),
        ("m2", "annual_fe_netmig_plus_origin_gdpcomp"),
        ("m3", "annual_fe_netmig_plus_origin_groupshares"),
        ("m4", "annual_fe_netmig_plus_gdpcomp_groupshares"),
    ]
    headers = [("m1", "Baseline"), ("m2", "GDP comp."), ("m3", "Groups"), ("m4", "Both")]
    rows: list[dict[str, Any]] = []
    for i, (term, label) in enumerate(
        [
            ("L1_net_migration_rate", "Lagged net migration rate"),
            ("L1_origin_gdp_pc_ppp_const_wavg_log", "Lagged weighted avg. origin GDPpc (log, PPP)"),
            ("L1_share_group_LATAM_CARIB", "Lagged share of inflows from Latin America/Caribbean"),
            ("L1_share_group_EU_EEA_CH_UK", "Lagged share of inflows from EU/EEA/CH/UK"),
            ("L1_share_group_MENA", "Lagged share of inflows from MENA"),
        ],
        start=1,
    ):
        add_coef_rows(rows, i, label, term, models, coef)
    add_midrule(rows, 90, [k for k, _ in headers])
    add_stat_row(rows, 91, "Country FE", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 92, "Year FE", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 93, "Macro + air controls", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 94, "Observations", {k: fmt_int(coef[coef["model"] == m]["nobs"].dropna().iloc[0]) for k, m in models})
    csv_path = write_long_csv("tab_who_arrives_composition", rows)
    p_row1 = coef_lookup(coef, "annual_fe_netmig_controls_matchedsample", "L1_net_migration_rate")
    p_row2a = coef_lookup(coef, "annual_fe_netmig_plus_origin_gdpcomp", "L1_net_migration_rate")
    p_row2b = coef_lookup(coef, "annual_fe_netmig_plus_origin_gdpcomp", "L1_origin_gdp_pc_ppp_const_wavg_log")
    tex = render_long_table(
        csv_path,
        caption="Who arrives matters: migration volume and inflow composition in country FE models",
        label="tab:who_arrives_composition",
        tabular_spec="p{5.0cm}p{2.1cm}p{2.1cm}p{2.1cm}p{2.1cm}",
        header_cols=headers,
        note_lines=[
            r"Notes: Dependent variable is annual house-price growth. All models include country and year fixed effects, lagged macro controls and lagged air controls, and clustered standard errors (country and year in estimation; table reports coefficient estimates and standard errors). The matched baseline uses the same country-year sample as the origin-GDP composition model. Origin-group shares are based on Eurostat origin-destination immigration inflows (\texttt{MIGR\_IMM5PRV}). Significance stars are computed directly from source-model p-values in the audited CSV pipeline (* $p<0.10$, ** $p<0.05$, *** $p<0.01$).",
            rf"Exact headline p-values (reported in text): matched-sample net migration coefficient $p={fmt_num(np.nan if p_row1 is None else p_row1['p_value'],3)}$ (Column 1); net migration + origin-GDP composition coefficient $p={fmt_num(np.nan if p_row2a is None else p_row2a['p_value'],3)}$ and origin-GDP composition coefficient $p={fmt_num(np.nan if p_row2b is None else p_row2b['p_value'],3)}$ (Column 2).",
        ],
        set_tabcolsep=4,
    )
    return csv_path, tex


def build_table_iv_results() -> tuple[Path, str]:
    coef = pd.read_csv(RESULTS_DIR / "advanced_model_coefficients.csv")
    models = [("m1", "country_iv_fe"), ("m2", "country_iv_fe_asylum")]
    headers = [("m1", "Composite shift-share IV"), ("m2", "Asylum shift-share IV")]
    rows: list[dict[str, Any]] = []
    for i, (term, label) in enumerate(
        [
            ("L1_net_migration_rate", "Lagged net migration rate (endogenous)"),
            ("L1_air_growth", "Lagged air-passenger growth"),
            ("L1_gdp_pc_growth", "Lagged GDP per-capita growth"),
            ("L1_unemployment_rate", "Lagged unemployment rate"),
        ],
        start=1,
    ):
        add_coef_rows(rows, i, label, term, models, coef)
    add_midrule(rows, 90, [k for k, _ in headers])
    add_stat_row(rows, 91, "Country FE", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 92, "Year FE", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 93, "Clustered SE (country)", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 94, "Observations", {k: fmt_int(coef[coef["model"] == m]["nobs"].dropna().iloc[0]) for k, m in models})
    csv_path = write_long_csv("tab_iv_results", rows)
    tex = render_long_table(
        csv_path,
        caption="Country-year IV estimates with country and year fixed effects",
        label="tab:iv_results",
        tabular_spec="p{6.0cm}cc",
        header_cols=headers,
        note_lines=[
            r"Notes: Dependent variable is annual house-price growth. The endogenous regressor is lagged net migration rate. Composite IV uses a World Bank push-shock shift-share exposure plus a leave-one-out origin-supply component; asylum IV uses a leave-one-out asylum-applications shift-share exposure. Significance stars are computed directly from source-model p-values in the audited CSV pipeline (* $p<0.10$, ** $p<0.05$, *** $p<0.01$).",
        ],
    )
    return csv_path, tex


def _fit_iv_for_diagnostics(instruments: list[str]) -> tuple[Any, pd.DataFrame]:
    d = pd.read_parquet(PROC_DIR / "panel_annual_harmonized.parquet").replace([np.inf, -np.inf], np.nan).copy()
    iv = pd.read_parquet(PROC_DIR / "panel_annual_iv.parquet").replace([np.inf, -np.inf], np.nan).copy()

    iv_cols = ["geo", "year"] + [c for c in instruments if c in iv.columns]
    d = d.merge(iv[iv_cols].drop_duplicates(["geo", "year"]), on=["geo", "year"], how="left")

    d["hpi_growth"] = pd.to_numeric(d.get("hpi_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d.get("hpi_growth"), errors="coerce")
    )
    d["net_migration_rate"] = pd.to_numeric(d.get("net_migration_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d.get("net_migration_rate"), errors="coerce")
    )
    d["air_growth"] = pd.to_numeric(d.get("air_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d.get("air_growth"), errors="coerce")
    )
    d["gdp_pc_growth"] = pd.to_numeric(d.get("gdp_pc_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d.get("gdp_pc_growth"), errors="coerce")
    )
    d["unemployment_rate"] = pd.to_numeric(d.get("unemployment_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d.get("unemployment_rate"), errors="coerce")
    )
    d["inflation_hicp"] = pd.to_numeric(d.get("inflation_hicp_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d.get("inflation_hicp"), errors="coerce")
    )
    d["long_rate"] = pd.to_numeric(d.get("long_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d.get("long_rate"), errors="coerce")
    )
    d["pop_growth"] = pd.to_numeric(d.get("pop_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(d.get("pop_growth"), errors="coerce")
    )

    d = d.sort_values(["geo", "year"]).reset_index(drop=True)
    g = d.groupby("geo", sort=False)
    for c in ["hpi_growth", "net_migration_rate", "air_growth", "gdp_pc_growth", "unemployment_rate", "inflation_hicp", "long_rate", "pop_growth"]:
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
    needed = [c for c in keep if c not in {"geo", "year"}]
    d = d.dropna(subset=needed).copy()
    formula = (
        "hpi_growth ~ 1 + L1_air_growth + L1_gdp_pc_growth + L1_unemployment_rate + "
        "L1_inflation_hicp + L1_long_rate + L1_pop_growth + C(geo) + C(year) "
        f"[L1_net_migration_rate ~ {' + '.join(instruments)}]"
    )
    res = IV2SLS.from_formula(formula, data=d).fit(cov_type="clustered", clusters=d["geo"])
    return res, d


def build_table_iv_diag() -> tuple[Path, str]:
    rows_csv: list[dict[str, Any]] = []
    specs = [
        ("Composite shift-share IV", ["L1_ss_push_index_wb", "L1_ss_loo_origin_supply_logdiff"], True),
        ("Asylum shift-share IV", ["L1_ss_asylum_loo_logdiff"], False),
    ]
    for i, (label, instrs, overid) in enumerate(specs, start=1):
        res, d = _fit_iv_for_diagnostics(instrs)
        diag = res.first_stage.diagnostics.iloc[0]
        sargan_stat = sargan_p = basmann_stat = basmann_p = np.nan
        if overid:
            sarg = getattr(res, "sargan", None)
            bas = getattr(res, "basmann", None)
            if sarg is not None:
                sargan_stat, sargan_p = float(sarg.stat), float(sarg.pval)
            if bas is not None:
                basmann_stat, basmann_p = float(bas.stat), float(bas.pval)
        vals = {
            "spec": label,
            "n_ivs": len(instrs),
            "partial_r2": float(diag.get("partial.rsquared", np.nan)),
            "partial_f": float(diag.get("f.stat", np.nan)),
            "f_pval": float(diag.get("f.pval", np.nan)),
            "sargan": sargan_stat,
            "sargan_p": sargan_p,
            "basmann": basmann_stat,
            "basmann_p": basmann_p,
            "n": int(len(d)),
        }
        vals["row_order"] = i
        rows_csv.append(vals)
    df = pd.DataFrame(rows_csv)
    csv_path = AUDIT_DIR / "tab_iv_diag.csv"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    body = []
    for _, r in df.sort_values("row_order").iterrows():
        body.append(
            " & ".join(
                [
                    latex_escape(r["spec"]),
                    fmt_int(r["n_ivs"]),
                    fmt_num(r["partial_r2"], 3),
                    fmt_num(r["partial_f"], 3),
                    fmt_num(r["f_pval"], 3),
                    fmt_num(r["sargan"], 3),
                    fmt_num(r["sargan_p"], 3),
                    fmt_num(r["basmann"], 3),
                    fmt_num(r["basmann_p"], 3),
                    fmt_int(r["n"]),
                ]
            )
            + r" \\"
        )
    tex = "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{IV first-stage and over-identification diagnostics}",
            r"\label{tab:iv_diag}",
            r"\begin{threeparttable}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{p{4.2cm}rrrrrrrrr}",
            r"\toprule",
            r"Specification & \# IVs & Partial $R^2$ & Partial F & F p-val & Sargan & Sargan p & Basmann & Basmann p & N \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item Notes: Diagnostics are extracted from the same 2SLS specifications used in Table~\ref{tab:iv_results} with country and year fixed effects and country-clustered covariance. Over-identification tests are only defined for the overidentified (two-instrument) specification.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    return csv_path, tex


def build_table_regional_fe() -> tuple[Path, str]:
    adv = pd.read_csv(RESULTS_DIR / "advanced_model_coefficients.csv")
    models = [("m1", "nuts2_twfe_baseline"), ("m2", "nuts2_twfe_post2020_interaction")]
    headers = [("m1", "NUTS2 TWFE baseline"), ("m2", "NUTS2 TWFE + post-2020 interactions")]
    rows: list[dict[str, Any]] = []
    for i, (term, label) in enumerate(
        [
            ("L1_net_migration_rate", "Lagged net migration rate"),
            ("L1_air_growth", "Lagged air-passenger growth"),
            ("L1_unemployment_rate", "Lagged unemployment rate"),
            ("L1_gdp_pc_growth", "Lagged GDP per-capita growth"),
            ("L1_net_migration_rate:post_2020", "Lagged migration × post-2020"),
            ("L1_air_growth:post_2020", "Lagged air growth × post-2020"),
        ],
        start=1,
    ):
        add_coef_rows(rows, i, label, term, models, adv)
    add_midrule(rows, 90, [k for k, _ in headers])
    add_stat_row(rows, 91, "Region FE", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 92, "Year FE", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 93, "Two-way clustered SE", {k: "Yes" for k, _ in headers})
    add_stat_row(rows, 94, "Observations", {k: fmt_int(adv[adv["model"] == m]["nobs"].dropna().iloc[0]) for k, m in models})
    csv_path = write_long_csv("tab_regional_fe", rows)
    tex = render_long_table(
        csv_path,
        caption="Exploratory NUTS2 annual fixed-effects estimates (corrected RHPI sample)",
        label="tab:regional_fe",
        tabular_spec="p{6.0cm}cc",
        header_cols=headers,
        note_lines=[
            r"Notes: Dependent variable is OECD regional RHPI annual growth. Sample shrinks materially after removing duplicate RHPI vintages and keeping total-stock (\texttt{VINTAGE=\_T}) observations only. One coefficient in the baseline model has an undefined clustered standard error due the small number of country clusters. Significance stars are computed directly from source-model p-values in the audited CSV pipeline (* $p<0.10$, ** $p<0.05$, *** $p<0.01$).",
        ],
        resize_to_textwidth=True,
    )
    return csv_path, tex


def build_table_event_pretrend() -> tuple[Path, str]:
    es_sum = json.loads((RESULTS_DIR / "event_study_pretrend_summary.json").read_text())
    label_map = {
        "regional_event_study_open_persistent": "Persistent route opening event",
        "regional_event_study_close_persistent": "Persistent route closure event",
    }
    rows = []
    for i, r in enumerate(es_sum, start=1):
        pt = r.get("pretrend_joint_test", {}) or {}
        rows.append(
            {
                "row_order": i,
                "event_definition": label_map.get(r["model"], r["model"]),
                "regions": int(r.get("regions_total", 0)),
                "treated": int(r.get("treated_regions", 0)),
                "obs": int(round(float(r.get("nobs", np.nan)))) if r.get("nobs") is not None else np.nan,
                "threshold": float(r.get("event_threshold_rate", np.nan)),
                "lead_df": int(pt.get("df", 0)) if pt.get("df") is not None else np.nan,
                "lead_chi2": float(pt.get("stat", np.nan)),
                "p_value": float(pt.get("p_value", np.nan)),
            }
        )
    df = pd.DataFrame(rows)
    csv_path = AUDIT_DIR / "tab_event_pretrend.csv"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    body = []
    for _, r in df.sort_values("row_order").iterrows():
        body.append(
            " & ".join(
                [
                    latex_escape(r["event_definition"]),
                    fmt_int(r["regions"]),
                    fmt_int(r["treated"]),
                    fmt_int(r["obs"]),
                    fmt_num(r["threshold"], 3),
                    fmt_int(r["lead_df"]),
                    fmt_num(r["lead_chi2"], 3),
                    fmt_num(r["p_value"], 3),
                ]
            )
            + r" \\"
        )
    tex = "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{Regional event-study pre-trend diagnostics (persistent route events)}",
            r"\label{tab:event_pretrend}",
            r"\begin{threeparttable}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{p{5.6cm}rrrrrrr}",
            r"\toprule",
            r"Event definition & Regions & Treated & Obs. & Threshold & Lead df & $\chi^2$ lead test & p-value \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item Notes: Event-study models use the first qualifying event per region, a symmetric event window of eight quarters before and after, omit event time $-1$, and include region and quarter fixed effects with two-way clustered standard errors.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    return csv_path, tex


def build_table_desc() -> tuple[Path, str]:
    s = pd.read_csv(RESULTS_DIR / "sample_stats.csv")
    q = pd.read_parquet(PROC_DIR / "panel_quarterly_harmonized.parquet")
    qoq = pd.to_numeric(q["dlog_hpi_qoq"], errors="coerce")
    qoq = qoq.replace([np.inf, -np.inf], np.nan).dropna()
    qoq_row = {
        "panel": "quarterly",
        "variable": "hpi_qoq_log_change_pp",
        "n": int(qoq.notna().sum()),
        "mean": float(qoq.mean()),
        "std": float(qoq.std(ddof=1)),
        "p10": float(qoq.quantile(0.10)),
        "p50": float(qoq.quantile(0.50)),
        "p90": float(qoq.quantile(0.90)),
    }
    rows_map = [
        ("annual", "hpi_growth", "Annual HPI growth (\\%)"),
        ("annual", "net_migration_rate", "Net migration rate (per 1,000)"),
        ("annual", "air_growth", "Air passenger growth (\\%)"),
        ("annual", "gdp_pc_growth", "GDP per capita growth (\\%)"),
        ("annual", "unemployment_rate", "Unemployment rate (\\%)"),
        ("annual", "inflation_hicp", "HICP inflation (\\%)"),
        ("quarterly", "hpi_yoy", "Quarterly HPI YoY growth (\\%)"),
        ("quarterly", "air_yoy", "Quarterly air passenger YoY growth (\\%)"),
    ]
    out_rows = []
    order = 1
    for panel, var, label in rows_map:
        r = s[(s["panel"] == panel) & (s["variable"] == var)]
        if r.empty:
            continue
        x = r.iloc[0]
        out_rows.append(
            {
                "row_order": order,
                "variable_label": label,
                "n": int(x["n"]),
                "mean": float(x["mean"]),
                "sd": float(x["std"]),
                "p10": float(x["p10"]),
                "median": float(x["p50"]),
                "p90": float(x["p90"]),
            }
        )
        order += 1
    out_rows.append(
        {
            "row_order": order,
            "variable_label": "Quarterly HPI QoQ log change (pp)",
            "n": qoq_row["n"],
            "mean": qoq_row["mean"],
            "sd": qoq_row["std"],
            "p10": qoq_row["p10"],
            "median": qoq_row["p50"],
            "p90": qoq_row["p90"],
        }
    )
    df = pd.DataFrame(out_rows)
    csv_path = AUDIT_DIR / "tab_desc.csv"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    body = []
    for _, r in df.sort_values("row_order").iterrows():
        body.append(
            " & ".join(
                [
                    r["variable_label"],
                    fmt_int(r["n"]),
                    f"{float(r['mean']):.2f}",
                    f"{float(r['sd']):.2f}",
                    f"{float(r['p10']):.2f}",
                    f"{float(r['median']):.2f}",
                    f"{float(r['p90']):.2f}",
                ]
            )
            + r" \\"
        )
    tex = "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{Descriptive statistics for main national analysis variables}",
            r"\label{tab:desc}",
            r"\begin{threeparttable}",
            r"\small",
            r"\begin{tabular}{p{5.5cm}rrrrrr}",
            r"\toprule",
            r"Variable & N & Mean & SD & P10 & Median & P90 \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item Notes: Statistics are computed from the processed annual and quarterly panels. Units follow the source definitions and transformations described in Section~\ref{sec:methodology}.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    return csv_path, tex


def build_table_coverage() -> tuple[Path, str]:
    panel = json.loads((META_DIR / "panel_build_summary.json").read_text())
    shift = json.loads((META_DIR / "shiftshare_iv_summary.json").read_text())
    flight = json.loads((META_DIR / "flight_shock_build_summary.json").read_text())
    cross = json.loads((META_DIR / "airport_nuts2_crosswalk_summary.json").read_text())
    rega = json.loads((META_DIR / "regional_panel_summary.json").read_text())
    regq = json.loads((META_DIR / "regional_route_quarterly_panel_summary.json").read_text())
    rows = [
        ("National annual panel", panel["annual_rows"], panel["annual_countries"], f"{panel['annual_year_min']}--{panel['annual_year_max']}", "Eurostat HPI + migration + air + macro controls"),
        ("National quarterly panel", panel["quarterly_rows"], panel["quarterly_countries"], f"{panel['quarterly_period_min']}--{panel['quarterly_period_max']}", "Eurostat HPI + air passengers"),
        ("Annual baseline estimation sample", panel["annual_full_sample_rows_for_baseline"], "", "2014--2024", "Non-missing HPI growth and lagged migration"),
        ("Quarterly baseline estimation sample", panel["quarterly_full_sample_rows_for_baseline"], "", "2006Q1--2025Q3", "Non-missing HPI YoY and lagged air YoY"),
        ("OD migration IV panel", shift["iv_rows"], shift["od_destinations"], f"{shift['od_year_min']}--{shift['od_year_max']}", "Destination-origin immigration exposures"),
        ("Monthly route-month panel", flight["route_month_rows"], 29, "2004--2025", "Airport-partner route activity"),
        ("Country-quarter route shocks", flight["country_quarter_rows"], flight["countries"], f"{flight['country_quarter_min']}--{flight['country_quarter_max']}", "Aggregated route openings/closures"),
        (
            "Airport-to-NUTS2 crosswalk",
            cross["airports_total"],
            "",
            "",
            f"Mapped {cross['nuts2_matched']}/{cross['airports_total']} airports ({100*cross['match_rate_nuts2']:.1f}\\%)",
        ),
        ("NUTS2 annual panel (corrected RHPI)", rega["rows"], rega["regions"], f"{rega['year_min']}--{rega['year_max']}", f"{rega['countries']} countries after RHPI vintage filter"),
        ("NUTS2 quarterly route-shock panel (corrected RHPI)", regq["rows"], regq["regions"], f"{regq['period_min']}--{regq['period_max']}", f"{regq['countries']} countries after RHPI vintage filter"),
    ]
    df = pd.DataFrame(rows, columns=["dataset_block", "rows", "units", "period", "notes"])
    df["row_order"] = np.arange(1, len(df) + 1)
    csv_path = AUDIT_DIR / "tab_coverage.csv"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    body = []
    for _, r in df.sort_values("row_order").iterrows():
        body.append(
            " & ".join(
                [
                    latex_escape(r["dataset_block"]),
                    fmt_int(r["rows"]),
                    "" if pd.isna(r["units"]) or r["units"] == "" else fmt_int(r["units"]),
                    latex_escape(r["period"]),
                    r["notes"],
                ]
            )
            + r" \\"
        )
    tex = "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{Dataset architecture and usable coverage}",
            r"\label{tab:coverage}",
            r"\begin{threeparttable}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{p{4.0cm}cccc}",
            r"\toprule",
            r"Dataset block & Rows & Units & Period & Notes \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item Notes: Regional OECD RHPI panels are filtered to total housing stock (VINTAGE = \texttt{\_T}) and non-seasonally adjusted observations (ADJUSTMENT = \texttt{N}) to avoid duplicate region-period observations.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    return csv_path, tex


def build_table_iberia_top_origins() -> tuple[Path, str]:
    top = pd.read_csv(RESULTS_DIR / "migration_case_es_pt_top_origins.csv")
    # Keep source-specific rank window measure if present; fallback to older column name.
    rank_col = "immig_total_rank_window" if "immig_total_rank_window" in top.columns else "immig_total_2018_2024"
    top_recent = (
        top.sort_values(["geo", rank_col], ascending=[True, False])
        .groupby("geo", as_index=False)
        .head(8)
        .copy()
    )
    top_recent["rank"] = top_recent.groupby("geo").cumcount() + 1
    csv_path = AUDIT_DIR / "tab_iberia_top_origins.csv"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    top_recent.to_csv(csv_path, index=False)

    def cell(df: pd.DataFrame, geo: str, rank: int) -> str:
        r = df[(df["geo"] == geo) & (df["rank"] == rank)]
        if r.empty:
            return ""
        x = r.iloc[0]
        tag = " [ctz]" if str(x.get("source", "")) == "ctz_proxy" else ""
        return f"{x['origin']} ({str(x['origin_group']).replace('_',' ')}, {fmt_int(x[rank_col])}){tag}"

    geos_order = [g for g in ["ES", "IT", "UK", "PT", "PL"] if g in set(top_recent["geo"].astype(str))]
    if not geos_order:
        geos_order = sorted(top_recent["geo"].astype(str).unique().tolist())

    def geo_name(geo: str) -> str:
        return {
            "ES": "Spain",
            "IT": "Italy",
            "UK": "United Kingdom",
            "PL": "Poland",
            "PT": "Portugal",
        }.get(geo, geo)

    def geo_header_label(df: pd.DataFrame, geo: str) -> str:
        sub = df[df["geo"] == geo]
        win = None
        if "rank_window_label" in sub.columns:
            w = sub["rank_window_label"].dropna().astype(str)
            if not w.empty:
                win = w.mode().iat[0]
        if not win:
            # Fallback for legacy files without rank-window labels.
            win = "2018--2024" if geo in {"ES", "IT"} else "country-specific OD window"
        source = None
        if "source" in sub.columns and not sub.empty:
            s = sub["source"].dropna().astype(str)
            if not s.empty:
                source = s.mode().iat[0]
        if source == "ctz_proxy":
            return f"{geo_name(geo)} ({geo}, citizenship proxy)"
        return f"{geo_name(geo)} ({geo}, OD {win})"

    body = []
    for rank in range(1, 9):
        row = [str(rank)]
        for geo in geos_order:
            row.append(cell(top_recent, geo, rank))
        body.append(" & ".join(row) + r" \\")

    colspec = "c " + " ".join(["p{4.1cm}"] * len(geos_order))
    header = "Rank & " + " & ".join(geo_header_label(top_recent, g) for g in geos_order) + r" \\"
    def join_names(xs: list[str]) -> str:
        if not xs:
            return "country cases"
        if len(xs) == 1:
            return xs[0]
        if len(xs) == 2:
            return f"{xs[0]} and {xs[1]}"
        return ", ".join(xs[:-1]) + f", and {xs[-1]}"

    case_names = [geo_name(g) for g in geos_order]
    note_line = (
        r"\item Notes: Entries show origin-country code (origin group, cumulative immigration counts in the country-specific ranking window shown in the header). "
        r"Baseline source is Eurostat \texttt{MIGR\_IMM5PRV} (previous residence, OD). Cells marked [ctz] use citizenship-proxy inflows "
        r"(\texttt{MIGR\_IMM1CTZ}); [ons] indicates UK ONS bridge-scaling; [blend] indicates multi-source year-level blending. "
        r"This table is descriptive and is used to visualize composition differences rather than to establish causal country rankings."
    )
    tex = "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            rf"\caption{{Country-case top country composition entries for immigration ({join_names(case_names)})}}",
            r"\label{tab:iberia_top_origins}",
            r"\begin{threeparttable}",
            r"\small",
            rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            note_line,
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    return csv_path, tex


def build_table_event_and_regional_from_sources() -> list[tuple[Path, str, str]]:
    out = []
    csv_path, tex = build_table_regional_fe()
    out.append((csv_path, "tab_regional_fe.tex", tex))
    csv_path, tex = build_table_event_pretrend()
    out.append((csv_path, "tab_event_pretrend.tex", tex))
    return out


def write_tex(filename: str, content: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / filename).write_text(content)


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    built: list[tuple[str, Path]] = []

    # Core regression and summary tables rebuilt from result/metadata CSVs and JSON.
    for fn, builder in [
        ("tab_baseline_annual.tex", build_table_baseline_annual),
        ("tab_quarterly.tex", build_table_quarterly),
        ("tab_traveler_quality_quarterly.tex", build_table_traveler_quality),
        ("tab_who_arrives_composition.tex", build_table_who_arrives),
        ("tab_iv_results.tex", build_table_iv_results),
        ("tab_iv_diag.tex", build_table_iv_diag),
        ("tab_desc.tex", build_table_desc),
        ("tab_coverage.tex", build_table_coverage),
        ("tab_iberia_top_origins.tex", build_table_iberia_top_origins),
    ]:
        csv_path, tex = builder()
        write_tex(fn, tex)
        built.append((fn, csv_path))

    for csv_path, fn, tex in build_table_event_and_regional_from_sources():
        write_tex(fn, tex)
        built.append((fn, csv_path))

    manifest = [
        {"tex": f"paper_overleaf/tables/{fn}", "csv": str(csv.relative_to(ROOT))}
        for fn, csv in built
    ]
    (AUDIT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[ok] rebuilt tables from audited CSV sources")
    for item in manifest:
        print(f"  {item['tex']} <- {item['csv']}")


if __name__ == "__main__":
    main()
