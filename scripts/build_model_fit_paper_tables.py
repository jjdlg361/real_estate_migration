#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
AUDIT_DIR = RESULTS_DIR / "paper_table_csv"
PAPER_TABLES_DIR = ROOT / "paper_overleaf" / "tables"


def fmt_num(x, digits=3):
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def fmt_pct(x, digits=1):
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}\\%"


def sample_label(s: str) -> str:
    return {
        "annual_headline_composition": "Annual headline (migration + origin-income composition)",
        "annual_extended_composition_groups": "Annual extended (add origin-region shares)",
        "quarterly_traveler_quality_full_interact": "Quarterly traveler-quality (full interacted)",
    }.get(s, s)


def factor_label(s: str) -> str:
    return {
        "migration": "Net migration block",
        "air": "Annual air-passenger growth",
        "macros": "Macro controls block",
        "origin_income_comp": "Origin-income composition",
        "origin_region_shares": "Origin-region shares",
        "air_growth": "Quarterly air-passenger growth",
        "route_shocks": "Route-shock controls",
        "airfare_proxy": "Airfare proxy (HICP CP0733)",
        "airline_mix_level": "Airline-licence mix proxy",
        "operating_intensity": "Operating intensity (passengers per movement)",
        "interactions": "Traveler-quality interactions",
    }.get(s, s.replace("_", " "))


def build_overview_table() -> tuple[pd.DataFrame, str]:
    seq = pd.read_csv(RESULTS_DIR / "model_fit_sequence.csv")
    rows = []
    for s, g in seq.groupby("sample", sort=False):
        g = g.sort_values("step")
        base = g.iloc[0]
        full = g.iloc[-1]
        rows.append(
            {
                "sample": s,
                "sample_label": sample_label(s),
                "nobs": int(full["nobs"]),
                "fe_only_rmse": float(base["rmse"]),
                "full_rmse": float(full["rmse"]),
                "rmse_impr_abs": float(base["rmse"] - full["rmse"]),
                "rmse_impr_pct": float((base["rmse"] - full["rmse"]) / base["rmse"] * 100.0) if base["rmse"] else np.nan,
                "fe_only_mae": float(base["mae"]),
                "full_mae": float(full["mae"]),
                "mae_impr_abs": float(base["mae"] - full["mae"]),
                "mae_impr_pct": float((base["mae"] - full["mae"]) / base["mae"] * 100.0) if base["mae"] else np.nan,
                "full_r2_inclusive": float(full["r2_inclusive"]),
            }
        )
    out = pd.DataFrame(rows)
    order = [
        "annual_headline_composition",
        "annual_extended_composition_groups",
        "quarterly_traveler_quality_full_interact",
    ]
    out["sample"] = pd.Categorical(out["sample"], categories=order, ordered=True)
    out = out.sort_values("sample").reset_index(drop=True)

    body = []
    for _, r in out.iterrows():
        body.append(
            " & ".join(
                [
                    r["sample_label"],
                    str(int(r["nobs"])),
                    fmt_num(r["fe_only_rmse"]),
                    fmt_num(r["full_rmse"]),
                    fmt_pct(r["rmse_impr_pct"]),
                    fmt_num(r["fe_only_mae"]),
                    fmt_num(r["full_mae"]),
                    fmt_pct(r["mae_impr_pct"]),
                    fmt_num(r["full_r2_inclusive"]),
                ]
            )
            + r" \\"
        )

    tex = "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{How much the full models improve fit: fixed-effects baseline versus full specifications}",
            r"\label{tab:model_fit_overview}",
            r"\small",
            r"\setlength{\tabcolsep}{3.8pt}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{p{5.8cm}cccccccc}",
            r"\toprule",
            r"Sample & Obs. & FE-only RMSE & Full RMSE & RMSE improvement & FE-only MAE & Full MAE & MAE improvement & Full $R^2$ (incl.) \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\vspace{2pt}",
            r"\parbox{0.98\textwidth}{\footnotesize \textit{Notes:} RMSE and MAE are in percentage points of house-price growth (annual growth for annual panels; quarterly YoY growth for the quarterly panel). ``FE-only'' means country and time fixed effects only (no covariate blocks). ``Full'' means the richest specification within each sample used in the fit-decomposition exercise. $R^2$ (incl.) is the inclusive model fit reported by the panel estimator and includes the contribution of fixed effects. These are in-sample fitted-value diagnostics, not true out-of-sample forecasts.}",
            r"\end{table}",
        ]
    )
    return out, tex


def build_block_table() -> tuple[pd.DataFrame, str]:
    lobo = pd.read_csv(RESULTS_DIR / "model_fit_leave_one_block_out.csv")
    # Keep all rows, sort by sample and contribution magnitude.
    order = [
        "annual_headline_composition",
        "annual_extended_composition_groups",
        "quarterly_traveler_quality_full_interact",
    ]
    lobo["sample"] = pd.Categorical(lobo["sample"], categories=order, ordered=True)
    lobo = lobo.sort_values(["sample", "delta_rmse"], ascending=[True, False]).reset_index(drop=True)
    lobo["sample_label"] = lobo["sample"].astype(str).map(sample_label)
    lobo["factor_label"] = lobo["factor"].astype(str).map(factor_label)

    # Human-readable table rows with group separators.
    body = []
    current_sample = None
    for _, r in lobo.iterrows():
        if r["sample"] != current_sample:
            if current_sample is not None:
                body.append(r"\addlinespace[2pt]")
            body.append(r"\multicolumn{6}{l}{\textit{" + str(r["sample_label"]).replace("&", r"\&") + r"}} \\")
            current_sample = r["sample"]
        body.append(
            " & ".join(
                [
                    "",
                    str(r["factor_label"]).replace("&", r"\&"),
                    fmt_num(r["delta_rmse"]),
                    fmt_num(r["delta_mae"]),
                    fmt_num(r["delta_r2_inclusive"]),
                    fmt_num(r["delta_r2_within"]),
                ]
            )
            + r" \\"
        )

    tex = "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{Which factors improve fit the most? Leave-one-block-out deterioration from the full model}",
            r"\label{tab:model_fit_blocks}",
            r"\begin{threeparttable}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{p{0.2cm}p{7.0cm}p{1.7cm}p{1.7cm}p{1.9cm}p{1.9cm}}",
            r"\toprule",
            r" & Factor block removed & $\Delta$RMSE & $\Delta$MAE & $\Delta R^2$ (incl.) & $\Delta R^2$ (within) \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item Notes: Each row compares the full model in a given sample to the same model after removing one factor block while keeping the fixed effects and all other blocks. Positive $\Delta$RMSE or $\Delta$MAE means the model gets worse when that block is removed, so larger positive values indicate more incremental explanatory content. Small negative values can appear when one error metric improves slightly while another worsens (common in noisy quarterly panels). $R^2$ changes are reported for both inclusive and within measures; the inclusive measure is usually easier to interpret in the presence of fixed effects.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    return lobo, tex


def main() -> None:
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    overview_df, overview_tex = build_overview_table()
    blocks_df, blocks_tex = build_block_table()

    overview_df.to_csv(AUDIT_DIR / "tab_model_fit_overview.csv", index=False)
    blocks_df.to_csv(AUDIT_DIR / "tab_model_fit_blocks.csv", index=False)
    (PAPER_TABLES_DIR / "tab_model_fit_overview.tex").write_text(overview_tex)
    (PAPER_TABLES_DIR / "tab_model_fit_blocks.tex").write_text(blocks_tex)

    print("[ok] wrote paper_overleaf/tables/tab_model_fit_overview.tex")
    print("[ok] wrote paper_overleaf/tables/tab_model_fit_blocks.tex")


if __name__ == "__main__":
    main()
