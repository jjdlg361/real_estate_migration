#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "paper_overleaf" / "figures"

IN_CSV = RESULTS_DIR / "next_year_house_price_projection_country.csv"
COEF_CSV = RESULTS_DIR / "dehaas_factors_coefficients.csv"
COEF_MODEL = "annual_fe_dehaas_core_interactions"
OUT_RANKED = FIG_DIR / "fig_country_next_year_projection_ranked.pdf"
OUT_SCATTER = FIG_DIR / "fig_country_next_year_projection_contrib_scatter.pdf"
OUT_SOCIO = FIG_DIR / "fig_country_next_year_projection_sociological_overlay.pdf"


def _country_name_map() -> dict[str, str]:
    return {
        "AT": "Austria",
        "BE": "Belgium",
        "BG": "Bulgaria",
        "CZ": "Czechia",
        "DE": "Germany",
        "DK": "Denmark",
        "EE": "Estonia",
        "ES": "Spain",
        "FI": "Finland",
        "FR": "France",
        "GR": "Greece",
        "HR": "Croatia",
        "HU": "Hungary",
        "IE": "Ireland",
        "IT": "Italy",
        "LT": "Lithuania",
        "LU": "Luxembourg",
        "LV": "Latvia",
        "NL": "Netherlands",
        "NO": "Norway",
        "PL": "Poland",
        "PT": "Portugal",
        "RO": "Romania",
        "SE": "Sweden",
        "SI": "Slovenia",
        "SK": "Slovakia",
        "UK": "United Kingdom",
    }


def plot_ranked(df: pd.DataFrame) -> None:
    d = df.sort_values("pred_hpi_growth_next_year_pp", ascending=True).copy()
    d["country_label"] = d["geo"].map(_country_name_map()).fillna(d["geo"])
    y = np.arange(len(d))
    vals = d["pred_hpi_growth_next_year_pp"].to_numpy()
    colors = np.where(vals >= 0, "#1f77b4", "#d62728")

    fig, ax = plt.subplots(figsize=(9.5, 10.5))
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.axvline(0.0, color="#444444", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(d["country_label"], fontsize=9)
    ax.set_xlabel("Predicted next-year house-price growth (percentage points)")
    ax.set_title("Country ranking: predicted annual house-price growth (t to t+1)")
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    for i, v in enumerate(vals):
        ha = "left" if v >= 0 else "right"
        x = v + (0.12 if v >= 0 else -0.12)
        ax.text(x, i, f"{v:.1f}", va="center", ha=ha, fontsize=8, color="#222222")

    fig.tight_layout()
    fig.savefig(OUT_RANKED, bbox_inches="tight")
    plt.close(fig)


def plot_contrib_scatter(df: pd.DataFrame) -> None:
    d = df.copy()
    x = pd.to_numeric(d["contrib_migration_system_pp"], errors="coerce")
    y = pd.to_numeric(d["contrib_macro_air_pp"], errors="coerce")
    c = pd.to_numeric(d["pred_hpi_growth_next_year_pp"], errors="coerce")
    s = 45 + 55 * np.clip(pd.to_numeric(d["max_predictor_staleness_years"], errors="coerce").fillna(0), 0, 6)

    fig, ax = plt.subplots(figsize=(10.2, 7.0))
    sc = ax.scatter(x, y, c=c, s=s, cmap="RdYlBu_r", alpha=0.9, edgecolor="#222222", linewidth=0.35)
    ax.axhline(0.0, color="#666666", lw=0.9)
    ax.axvline(0.0, color="#666666", lw=0.9)
    ax.set_xlabel("Contribution from migration-system block (pp)")
    ax.set_ylabel("Contribution from macro + air block (pp)")
    ax.set_title("What drives predicted next-year growth by country")
    ax.grid(True, linestyle="--", alpha=0.2)

    for _, r in d.iterrows():
        ax.text(
            float(r["contrib_migration_system_pp"]) + 0.05,
            float(r["contrib_macro_air_pp"]) + 0.05,
            str(r["geo"]),
            fontsize=8,
            color="#222222",
        )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.01)
    cbar.set_label("Predicted next-year growth (pp)")
    fig.tight_layout()
    fig.savefig(OUT_SCATTER, bbox_inches="tight")
    plt.close(fig)


def plot_sociological_overlay(df: pd.DataFrame) -> None:
    coef = pd.read_csv(COEF_CSV)
    mc = coef[coef["model"] == COEF_MODEL].copy()
    mc["p_value"] = pd.to_numeric(mc["p_value"], errors="coerce")
    sig_terms = set(mc.loc[mc["p_value"] < 0.10, "term"].astype(str))

    term_group = {
        "L1_net_migration_rate": "Migration volume",
        "L1_origin_income_hump": "Origin composition",
        "L1_network_stock_per_1000": "Network depth",
        "L1_network_hhi": "Origin concentration",
        "L1_asylum_share_inflows": "Legal channel mix",
        "L1_gdp_pc_growth": "Macro growth",
        "L1_unemployment_rate": "Labor market slack",
        "L1_inflation_hicp": "Price pressure",
    }
    ordered_groups = [
        "Migration volume",
        "Origin composition",
        "Network depth",
        "Origin concentration",
        "Legal channel mix",
        "Macro growth",
        "Labor market slack",
        "Price pressure",
    ]
    group_colors = {
        "Migration volume": "#4c78a8",
        "Origin composition": "#f58518",
        "Network depth": "#54a24b",
        "Origin concentration": "#e45756",
        "Legal channel mix": "#b279a2",
        "Macro growth": "#72b7b2",
        "Labor market slack": "#ff9da6",
        "Price pressure": "#9d755d",
    }

    d = df.copy()
    for g in ordered_groups:
        d[g] = 0.0
    for term, grp in term_group.items():
        ccol = f"contrib__{term}"
        if ccol not in d.columns:
            continue
        if term not in sig_terms:
            continue
        d[grp] = d[grp] + pd.to_numeric(d[ccol], errors="coerce").fillna(0.0)

    d = d.sort_values("pred_hpi_growth_next_year_pp", ascending=True).copy()
    d["country_label"] = d["geo"].map(_country_name_map()).fillna(d["geo"])
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(11.5, 11.0))
    left_pos = np.zeros(len(d))
    left_neg = np.zeros(len(d))
    for g in ordered_groups:
        vals = pd.to_numeric(d[g], errors="coerce").fillna(0.0).to_numpy()
        pos = np.clip(vals, 0, None)
        neg = np.clip(vals, None, 0)
        if np.any(np.abs(pos) > 0):
            ax.barh(y, pos, left=left_pos, color=group_colors[g], edgecolor="white", linewidth=0.3, label=g)
            left_pos = left_pos + pos
        if np.any(np.abs(neg) > 0):
            ax.barh(y, neg, left=left_neg, color=group_colors[g], edgecolor="white", linewidth=0.3)
            left_neg = left_neg + neg

    pred = pd.to_numeric(d["pred_hpi_growth_next_year_pp"], errors="coerce").to_numpy()
    ax.scatter(pred, y, marker="D", s=26, color="#111111", label="Model prediction (full)")

    ax.axvline(0.0, color="#444444", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(d["country_label"], fontsize=9)
    ax.set_xlabel("Contribution to predicted next-year house-price growth (pp)")
    ax.set_title("Sociological overlay: significant driver blocks vs full prediction")
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    # Force a complete legend even when a block is only negative (or zero) in the sample.
    legend_handles = [Patch(facecolor=group_colors[g], edgecolor="white", label=g) for g in ordered_groups]
    legend_handles.append(Line2D([0], [0], marker="D", color="#111111", linestyle="None", markersize=5, label="Model prediction (full)"))
    ax.legend(handles=legend_handles, loc="lower right", ncol=2, fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(OUT_SOCIO, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    needed = {"geo", "pred_hpi_growth_next_year_pp", "contrib_migration_system_pp", "contrib_macro_air_pp"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {IN_CSV}: {missing}")
    plot_ranked(df)
    plot_contrib_scatter(df)
    plot_sociological_overlay(df)
    print(f"[ok] wrote {OUT_RANKED}")
    print(f"[ok] wrote {OUT_SCATTER}")
    print(f"[ok] wrote {OUT_SOCIO}")


if __name__ == "__main__":
    main()
