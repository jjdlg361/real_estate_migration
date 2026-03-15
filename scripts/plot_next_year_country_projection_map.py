#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
PAPER_FIGS_DIR = ROOT / "paper_overleaf" / "figures"

MODEL = "annual_fe_dehaas_core_interactions"
COEF_FILE = RESULTS_DIR / "dehaas_factors_coefficients.csv"
PANEL_FILE = PROC_DIR / "panel_annual_dehaas_factors.parquet"
OUT_CSV = RESULTS_DIR / "next_year_house_price_projection_country.csv"
OUT_FIG = PAPER_FIGS_DIR / "fig_country_next_year_projection_map.pdf"
OUT_META = RESULTS_DIR / "next_year_house_price_projection_meta.json"
OUT_AUDIT = RESULTS_DIR / "next_year_house_price_projection_input_audit.csv"

# User-request horizon: Mar-2026 to Mar-2027.
TARGET_YEAR = 2027

# If 2026 covariates are unavailable, use latest observed covariates as proxy.
PREFERRED_SOURCE_YEAR = 2026

# Projection guardrails for as-of-now inputs.
STRICT_NO_WB_FALLBACK = True
MAX_COMPOSITION_STALENESS_YEARS = 2


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
    path = RAW_DIR / "gisco_nuts2_2021_4326.geojson"
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
    return {k: unary_union(v) for k, v in by_country.items() if v}


def _draw_country_choropleth(
    ax,
    geoms: dict[str, object],
    values: pd.Series,
    title: str,
    cmap: str,
    diverging: bool,
    value_label: str,
) -> None:
    vals = values.dropna()
    countries = [c for c in vals.index if c in geoms]
    if not countries:
        ax.set_axis_off()
        ax.set_title(title)
        return

    v = vals.loc[countries]
    if diverging:
        vmax = float(np.nanmax(np.abs(v.values))) if len(v) else 1.0
        vmax = max(vmax, 1e-6)
        vmin = -vmax
    else:
        vmin = float(np.nanmin(v.values))
        vmax = float(np.nanmax(v.values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6

    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    patches = []
    colors = []
    for c in countries:
        for coords in _iter_exterior_coords(geoms[c]):
            patches.append(MplPolygon(coords[:, :2], closed=True))
            colors.append(vals.loc[c])
    if patches:
        pc = PatchCollection(patches, cmap=cmap_obj, norm=norm, linewidths=0.35, edgecolor="#FFFFFF")
        pc.set_array(np.asarray(colors))
        ax.add_collection(pc)
        cbar = plt.colorbar(pc, ax=ax, fraction=0.04, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(value_label, fontsize=8)

    missing = [c for c in geoms if c not in countries]
    miss = []
    for c in missing:
        for coords in _iter_exterior_coords(geoms[c]):
            miss.append(MplPolygon(coords[:, :2], closed=True))
    if miss:
        pc_m = PatchCollection(miss, facecolor="#F1F1F1", edgecolor="#DDDDDD", linewidths=0.25)
        ax.add_collection(pc_m)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-25, 40)
    ax.set_ylim(34, 72)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def _latest_panel_year(df: pd.DataFrame) -> int:
    yrs = pd.to_numeric(df["year"], errors="coerce").dropna()
    if yrs.empty:
        raise RuntimeError("Panel has no usable year values.")
    return int(yrs.max())


def _build_country_snapshot(
    panel: pd.DataFrame,
    source_year: int,
    required_cols: list[str],
    source_col_by_predictor: dict[str, str] | None = None,
    blocked_sources_by_predictor: dict[str, set[str]] | None = None,
    enforce_blocked_sources: bool = False,
) -> pd.DataFrame:
    """
    Build one row per country using the latest available value for each predictor
    at or before source_year (country-specific carry-forward).
    """
    w = panel[pd.to_numeric(panel["year"], errors="coerce") <= source_year].copy()
    w = w.sort_values(["geo", "year"])
    rows: list[dict] = []
    for geo, g in w.groupby("geo", sort=False):
        rec: dict[str, float | int | str] = {"geo": str(geo)}
        age_by_col: list[int] = []
        for c in required_cols:
            src_col = (source_col_by_predictor or {}).get(c)
            use_cols = ["year", c] + ([src_col] if src_col and src_col in g.columns else [])
            z = g[use_cols].dropna(subset=[c])
            blocked = (blocked_sources_by_predictor or {}).get(c, set())
            if src_col and src_col in z.columns and blocked:
                z_pref = z[~z[src_col].astype(str).isin(blocked)].copy()
                if not z_pref.empty:
                    z = z_pref
                elif enforce_blocked_sources:
                    z = z_pref
            if z.empty:
                rec[c] = np.nan
                continue
            yr = int(pd.to_numeric(z["year"], errors="coerce").iloc[-1])
            rec[c] = float(pd.to_numeric(z[c], errors="coerce").iloc[-1])
            rec[f"{c}_source_year"] = yr
            age_by_col.append(max(0, source_year - yr))
        rec["max_predictor_staleness_years"] = int(max(age_by_col)) if age_by_col else np.nan
        rec["avg_predictor_staleness_years"] = float(np.mean(age_by_col)) if age_by_col else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PANEL_FILE).replace([np.inf, -np.inf], np.nan)
    coef = pd.read_csv(COEF_FILE)
    mcoef = coef[coef["model"] == MODEL].copy()
    if mcoef.empty:
        raise RuntimeError(f"No coefficients found for model={MODEL}")

    # term -> source-year variable (unlagged, because we project t+1 from year t information)
    term_var_map = {
        "L1_net_migration_rate": "net_migration_rate",
        "L1_origin_income_hump": "origin_income_hump",
        "L1_network_stock_per_1000": "network_stock_per_1000",
        "L1_network_hhi": "network_hhi",
        "L1_asylum_share_inflows": "asylum_share_inflows",
        "L1_naturalization_rate_per_1000": "naturalization_rate_per_1000",
        "L1_air_growth": "air_growth",
        "L1_gdp_pc_growth": "gdp_pc_growth",
        "L1_unemployment_rate": "unemployment_rate",
        "L1_inflation_hicp": "inflation_hicp",
        "L1_long_rate": "long_rate",
        "L1_pop_growth": "pop_growth",
    }
    interaction_term = "L1_net_migration_rate:L1_origin_income_hump"

    required_source_cols = sorted(set(term_var_map.values()))
    latest_year = _latest_panel_year(panel)
    source_year = min(PREFERRED_SOURCE_YEAR, latest_year)

    source_col_by_predictor = {
        "net_migration_rate": "net_migration_rate_harmonized_source",
    }
    blocked_sources_by_predictor = {
        # Prefer observed panel/official harmonized sources over WB fallback for projection snapshots.
        "net_migration_rate": {"wb_net_migration"},
    }
    proj = _build_country_snapshot(
        panel,
        source_year=source_year,
        required_cols=required_source_cols,
        source_col_by_predictor=source_col_by_predictor,
        blocked_sources_by_predictor=blocked_sources_by_predictor,
        enforce_blocked_sources=STRICT_NO_WB_FALLBACK,
    )

    # Composition variables can be much older for some countries; neutralize very stale
    # values to the cross-country median to avoid stale extremes dominating the map.
    comp_vars = [
        "origin_income_hump",
        "network_stock_per_1000",
        "network_hhi",
        "asylum_share_inflows",
        "naturalization_rate_per_1000",
    ]
    stale_replacements: dict[str, int] = {}
    for c in comp_vars:
        year_col = f"{c}_source_year"
        if c not in proj.columns or year_col not in proj.columns:
            continue
        age = source_year - pd.to_numeric(proj[year_col], errors="coerce")
        stale_mask = age > MAX_COMPOSITION_STALENESS_YEARS
        if stale_mask.any():
            med = pd.to_numeric(proj[c], errors="coerce").median()
            proj.loc[stale_mask, c] = med
            stale_replacements[c] = int(stale_mask.sum())

    proj = proj.dropna(subset=required_source_cols).copy()
    if proj.empty:
        raise RuntimeError(f"No complete rows for source year {source_year}.")

    # Match training preprocessing: clip using historical lag-term 1%/99% bounds.
    for term, src in term_var_map.items():
        if term not in panel.columns:
            proj[f"{src}_clip"] = pd.to_numeric(proj[src], errors="coerce")
            continue
        hist = pd.to_numeric(panel[term], errors="coerce").dropna()
        if hist.empty:
            proj[f"{src}_clip"] = pd.to_numeric(proj[src], errors="coerce")
            continue
        lo, hi = hist.quantile([0.01, 0.99])
        proj[f"{src}_clip"] = pd.to_numeric(proj[src], errors="coerce").clip(lo, hi)

    beta = dict(zip(mcoef["term"].astype(str), pd.to_numeric(mcoef["coef"], errors="coerce")))
    intercept = float(beta.get("Intercept", 0.0))

    # Build term values used in projection.
    tvals = pd.DataFrame({"geo": proj["geo"].astype(str)})
    for term, src in term_var_map.items():
        tvals[term] = pd.to_numeric(proj[f"{src}_clip"], errors="coerce")
    tvals[interaction_term] = tvals["L1_net_migration_rate"] * tvals["L1_origin_income_hump"]

    # Per-term contribution in percentage points.
    contrib_cols = []
    for term in term_var_map:
        ccol = f"contrib__{term}"
        tvals[ccol] = float(beta.get(term, 0.0)) * pd.to_numeric(tvals[term], errors="coerce")
        contrib_cols.append(ccol)
    tvals[f"contrib__{interaction_term}"] = float(beta.get(interaction_term, 0.0)) * pd.to_numeric(
        tvals[interaction_term], errors="coerce"
    )
    contrib_cols.append(f"contrib__{interaction_term}")

    tvals["pred_hpi_growth_next_year_pp"] = intercept + tvals[contrib_cols].sum(axis=1, min_count=1)

    migration_terms = [
        "L1_net_migration_rate",
        "L1_origin_income_hump",
        "L1_network_stock_per_1000",
        "L1_network_hhi",
        "L1_asylum_share_inflows",
        "L1_naturalization_rate_per_1000",
        interaction_term,
    ]
    macro_terms = [
        "L1_air_growth",
        "L1_gdp_pc_growth",
        "L1_unemployment_rate",
        "L1_inflation_hicp",
        "L1_long_rate",
        "L1_pop_growth",
    ]
    tvals["contrib_migration_system_pp"] = tvals[[f"contrib__{x}" for x in migration_terms]].sum(axis=1, min_count=1)
    tvals["contrib_macro_air_pp"] = tvals[[f"contrib__{x}" for x in macro_terms]].sum(axis=1, min_count=1)
    source_year_cols = [f"{src}_source_year" for src in required_source_cols if f"{src}_source_year" in proj.columns]
    if source_year_cols:
        sy = proj[source_year_cols].copy()
        tvals["last_input_data_year"] = sy.max(axis=1, skipna=True)
        tvals["earliest_input_data_year"] = sy.min(axis=1, skipna=True)
    else:
        tvals["last_input_data_year"] = np.nan
        tvals["earliest_input_data_year"] = np.nan
    tvals["max_predictor_staleness_years"] = pd.to_numeric(proj["max_predictor_staleness_years"], errors="coerce")
    tvals["avg_predictor_staleness_years"] = pd.to_numeric(proj["avg_predictor_staleness_years"], errors="coerce")
    tvals["projection_target_year"] = TARGET_YEAR
    tvals["projection_source_year"] = source_year
    tvals["model"] = MODEL
    tvals["assumption"] = (
        f"Country-specific predictor snapshot as of {source_year}: each regressor uses latest available value at or before {source_year}; "
        f"used as t-1 proxy inputs for target year {TARGET_YEAR}."
    )

    out = tvals.sort_values("pred_hpi_growth_next_year_pp", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    audit_cols = ["geo"]
    for src in required_source_cols:
        if src in proj.columns:
            audit_cols.append(src)
        yc = f"{src}_source_year"
        if yc in proj.columns:
            audit_cols.append(yc)
    if "max_predictor_staleness_years" in proj.columns:
        audit_cols.append("max_predictor_staleness_years")
    if "avg_predictor_staleness_years" in proj.columns:
        audit_cols.append("avg_predictor_staleness_years")
    proj[audit_cols].sort_values("geo").to_csv(OUT_AUDIT, index=False)

    geoms = load_country_polygons()
    if geoms:
        val_pred = out.set_index("geo")["pred_hpi_growth_next_year_pp"]
        val_ms = out.set_index("geo")["contrib_migration_system_pp"]

        fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), constrained_layout=True)
        _draw_country_choropleth(
            axes[0],
            geoms,
            val_pred,
            title=f"Model-implied next-year HPI growth (pp)\nTarget: {TARGET_YEAR} (source covariates: {source_year})",
            cmap="YlOrRd",
            diverging=False,
            value_label="Predicted annual growth (pp)",
        )
        _draw_country_choropleth(
            axes[1],
            geoms,
            val_ms,
            title="Migration-system block contribution (pp)",
            cmap="RdBu_r",
            diverging=True,
            value_label="Contribution (pp)",
        )
        fig.suptitle(
            "Country projection map: model-implied next-year house-price growth and migration-system contribution",
            fontsize=12,
            y=1.02,
        )
        fig.savefig(OUT_FIG, bbox_inches="tight")
        plt.close(fig)

    meta = {
        "model": MODEL,
        "target_year": TARGET_YEAR,
        "source_year": source_year,
        "n_countries_projected": int(out["geo"].nunique()),
        "latest_input_data_year_used_max": float(pd.to_numeric(out["last_input_data_year"], errors="coerce").max()),
        "latest_input_data_year_used_min": float(pd.to_numeric(out["last_input_data_year"], errors="coerce").min()),
        "strict_no_wb_fallback": bool(STRICT_NO_WB_FALLBACK),
        "strict_no_wb_fallback_predictors": ["net_migration_rate"],
        "max_composition_staleness_years": int(MAX_COMPOSITION_STALENESS_YEARS),
        "stale_composition_replacements": stale_replacements,
        "max_predictor_staleness_years": float(pd.to_numeric(out["max_predictor_staleness_years"], errors="coerce").max()),
        "avg_predictor_staleness_years": float(pd.to_numeric(out["avg_predictor_staleness_years"], errors="coerce").mean()),
        "predicted_growth_min_pp": float(out["pred_hpi_growth_next_year_pp"].min()),
        "predicted_growth_max_pp": float(out["pred_hpi_growth_next_year_pp"].max()),
        "outputs": {"csv": str(OUT_CSV), "figure": str(OUT_FIG)},
    }
    OUT_META.write_text(json.dumps(meta, indent=2))

    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_AUDIT}")
    print(f"[ok] wrote {OUT_FIG}")
    print(f"[ok] wrote {OUT_META}")


if __name__ == "__main__":
    main()
