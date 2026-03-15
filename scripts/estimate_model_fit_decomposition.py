#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import math
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"linearmodels(\..*)?")

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
META_DIR = ROOT / "data" / "metadata"


def winsorize_series(s: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.quantile([p_low, p_high])
    return x.clip(lo, hi)


def _predict_with_effects(res) -> pd.Series:
    pr = res.predict(effects=True)
    yhat = pr["fitted_values"].copy()
    if "estimated_effects" in pr.columns:
        yhat = yhat + pr["estimated_effects"]
    yhat.name = "yhat"
    return yhat


def _fit_metrics(
    df_panel: pd.DataFrame,
    y: str,
    terms: list[str],
    cluster_entity: bool = True,
    cluster_time: bool = True,
) -> dict:
    rhs = " + ".join(terms) if terms else "1"
    formula = f"{y} ~ 1" + (f" + {' + '.join(terms)}" if terms else "") + " + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(formula, data=df_panel, drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type="clustered", cluster_entity=cluster_entity, cluster_time=cluster_time)

    y_true = pd.to_numeric(df_panel[y], errors="coerce")
    yhat = _predict_with_effects(res).reindex(y_true.index)
    err = y_true - yhat

    rmse = float(np.sqrt(np.nanmean(np.square(err)))) if err.notna().any() else np.nan
    mae = float(np.nanmean(np.abs(err))) if err.notna().any() else np.nan
    corr = float(pd.concat([y_true.rename("y"), yhat], axis=1).corr().iloc[0, 1]) if len(y_true) > 1 else np.nan

    out = {
        "formula": formula,
        "nobs": int(res.nobs),
        "n_entities": int(df_panel.index.get_level_values(0).nunique()),
        "n_periods": int(df_panel.index.get_level_values(1).nunique()),
        "r2": float(getattr(res, "rsquared", np.nan)),
        "r2_inclusive": float(getattr(res, "rsquared_inclusive", np.nan)),
        "r2_within": float(getattr(res, "rsquared_within", np.nan)),
        "r2_overall": float(getattr(res, "rsquared_overall", np.nan)),
        "rmse": rmse,
        "mae": mae,
        "corr_actual_fitted": corr,
        "result": res,
    }
    return out


def _all_required_cols(blocks: OrderedDict[str, dict], y: str) -> list[str]:
    cols = [y]
    for spec in blocks.values():
        cols.extend(spec.get("required", []))
    # preserve order
    out = []
    seen = set()
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _subset_terms(blocks: OrderedDict[str, dict], subset: tuple[str, ...]) -> list[str]:
    terms: list[str] = []
    for b in subset:
        terms.extend(blocks[b]["terms"])
    return terms


def _factorial(n: int) -> int:
    return math.factorial(n)


def evaluate_block_decomposition(
    df_raw: pd.DataFrame,
    *,
    sample_name: str,
    y: str,
    index_cols: tuple[str, str],
    blocks: OrderedDict[str, dict],
    sequence: list[str],
    cluster_time: bool = True,
    winsorize_cols: list[str] | None = None,
    preprocessor=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    d = df_raw.replace([np.inf, -np.inf], np.nan).copy()
    if preprocessor is not None:
        d = preprocessor(d)
    if winsorize_cols:
        for c in winsorize_cols:
            if c in d.columns:
                d[c] = winsorize_series(d[c], 0.01, 0.99)

    needed = [c for c in _all_required_cols(blocks, y) if c in d.columns]
    d = d.dropna(subset=needed).copy()
    d[index_cols[1]] = pd.to_numeric(d[index_cols[1]], errors="coerce")
    d = d.dropna(subset=[index_cols[0], index_cols[1]])
    d[index_cols[1]] = d[index_cols[1]].astype(int)
    d = d.set_index(list(index_cols)).sort_index()

    block_names = list(blocks.keys())
    cache: dict[tuple[str, ...], dict] = {}

    def fit_subset(subset: tuple[str, ...]) -> dict:
        key = tuple(sorted(subset))
        if key in cache:
            return cache[key]
        terms = _subset_terms(blocks, key)
        fit = _fit_metrics(d, y, terms, cluster_entity=True, cluster_time=cluster_time)
        fit["subset"] = key
        cache[key] = fit
        return fit

    # Fit all subsets once (2^K models).
    for r in range(len(block_names) + 1):
        for subset in itertools.combinations(block_names, r):
            fit_subset(subset)

    # Sequential build-up
    seq_rows = []
    prev_key: tuple[str, ...] = tuple()
    prev_fit = cache[prev_key]
    seq_rows.append(
        {
            "sample": sample_name,
            "step": 0,
            "model_label": "FE only",
            "blocks_included": "",
            "added_block": "",
            **{k: prev_fit[k] for k in ["nobs", "n_entities", "n_periods", "r2", "r2_inclusive", "r2_within", "r2_overall", "rmse", "mae", "corr_actual_fitted"]},
            "delta_r2_inclusive": np.nan,
            "delta_r2_within": np.nan,
            "delta_rmse": np.nan,
            "delta_mae": np.nan,
        }
    )
    current: list[str] = []
    for i, block in enumerate(sequence, start=1):
        current.append(block)
        key = tuple(sorted(current))
        fit = cache[key]
        seq_rows.append(
            {
                "sample": sample_name,
                "step": i,
                "model_label": "FE + " + " + ".join(current),
                "blocks_included": "|".join(current),
                "added_block": block,
                **{k: fit[k] for k in ["nobs", "n_entities", "n_periods", "r2", "r2_inclusive", "r2_within", "r2_overall", "rmse", "mae", "corr_actual_fitted"]},
                "delta_r2_inclusive": fit["r2_inclusive"] - prev_fit["r2_inclusive"],
                "delta_r2_within": fit["r2_within"] - prev_fit["r2_within"],
                "delta_rmse": fit["rmse"] - prev_fit["rmse"],
                "delta_mae": fit["mae"] - prev_fit["mae"],
            }
        )
        prev_fit = fit
    seq_df = pd.DataFrame(seq_rows)

    # Leave-one-block-out vs full model
    full_key = tuple(sorted(block_names))
    full_fit = cache[full_key]
    lobo_rows = []
    for b in block_names:
        reduced_key = tuple(sorted([x for x in block_names if x != b]))
        reduced_fit = cache[reduced_key]
        lobo_rows.append(
            {
                "sample": sample_name,
                "factor": b,
                "comparison": "full_minus_without_block",
                "full_r2_inclusive": full_fit["r2_inclusive"],
                "full_r2_within": full_fit["r2_within"],
                "full_rmse": full_fit["rmse"],
                "full_mae": full_fit["mae"],
                "without_block_r2_inclusive": reduced_fit["r2_inclusive"],
                "without_block_r2_within": reduced_fit["r2_within"],
                "without_block_rmse": reduced_fit["rmse"],
                "without_block_mae": reduced_fit["mae"],
                "delta_r2_inclusive": full_fit["r2_inclusive"] - reduced_fit["r2_inclusive"],
                "delta_r2_within": full_fit["r2_within"] - reduced_fit["r2_within"],
                "delta_rmse": reduced_fit["rmse"] - full_fit["rmse"],  # positive = full improves RMSE
                "delta_mae": reduced_fit["mae"] - full_fit["mae"],      # positive = full improves MAE
            }
        )
    lobo_df = pd.DataFrame(lobo_rows)

    # Shapley decomposition for within-R2 and inclusive-R2
    n = len(block_names)
    shap_rows = []
    for b in block_names:
        phi_within = 0.0
        phi_incl = 0.0
        others = [x for x in block_names if x != b]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                s_key = tuple(sorted(subset))
                sb_key = tuple(sorted(subset + (b,)))
                w = _factorial(len(subset)) * _factorial(n - len(subset) - 1) / _factorial(n)
                phi_within += w * (cache[sb_key]["r2_within"] - cache[s_key]["r2_within"])
                phi_incl += w * (cache[sb_key]["r2_inclusive"] - cache[s_key]["r2_inclusive"])
        shap_rows.append(
            {
                "sample": sample_name,
                "factor": b,
                "comparison": "shapley",
                "shapley_r2_within": phi_within,
                "shapley_r2_inclusive": phi_incl,
            }
        )
    shap_df = pd.DataFrame(shap_rows)

    meta = {
        "sample": sample_name,
        "rows_common_sample": int(len(d)),
        "entities_common_sample": int(d.index.get_level_values(0).nunique()),
        "periods_common_sample": int(d.index.get_level_values(1).nunique()),
        "blocks": block_names,
        "sequence": sequence,
        "full_model": {
            "r2_inclusive": float(full_fit["r2_inclusive"]),
            "r2_within": float(full_fit["r2_within"]),
            "rmse": float(full_fit["rmse"]),
            "mae": float(full_fit["mae"]),
            "corr_actual_fitted": float(full_fit["corr_actual_fitted"]),
            "formula": full_fit["formula"],
        },
    }
    return seq_df, lobo_df, shap_df, meta


def prep_annual_comp(df: pd.DataFrame) -> pd.DataFrame:
    # Mirror the composition script transformations used for estimation.
    for col in [
        "hpi_growth",
        "L1_immigration_rate_per_1000",
        "L1_net_migration_rate",
        "L1_air_growth",
        "L1_origin_gdp_pc_ppp_const_wavg_log",
        "L1_origin_gdp_pc_ppp_const_wavg_log_c",
    ]:
        if col in df.columns:
            df[col] = winsorize_series(df[col], 0.01, 0.99)
    return df


def prep_quarterly_tq(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "hpi_yoy", "L1_air_yoy", "L1_open_rate_norm_q", "L1_close_rate_norm_q",
        "L1_airfare_yoy_q", "L1_lic_neu_share_pas", "L1_pax_per_move_total",
    ]:
        if col in df.columns:
            df[col] = winsorize_series(df[col], 0.01, 0.99)
    if "L1_lic_neu_share_pas" in df.columns and "L1_lic_neu_share_pas_c" not in df.columns:
        df["L1_lic_neu_share_pas_c"] = df["L1_lic_neu_share_pas"] - df["L1_lic_neu_share_pas"].mean(skipna=True)
    if "L1_airfare_yoy_q" in df.columns and "L1_airfare_yoy_q_c" not in df.columns:
        df["L1_airfare_yoy_q_c"] = df["L1_airfare_yoy_q"] - df["L1_airfare_yoy_q"].mean(skipna=True)
    return df


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    annual = pd.read_parquet(PROC_DIR / "panel_annual_migration_composition.parquet")
    quarterly = pd.read_parquet(PROC_DIR / "panel_quarterly_traveler_quality.parquet")

    # Annual headline "who arrives" sample (matches Table 6 Column 2 sample logic, no region-share requirement).
    annual_headline_blocks = OrderedDict(
        [
            ("migration", {"terms": ["L1_net_migration_rate"], "required": ["L1_net_migration_rate"]}),
            ("air", {"terms": ["L1_air_growth"], "required": ["L1_air_growth"]}),
            (
                "macros",
                {
                    "terms": [
                        "L1_gdp_pc_growth",
                        "L1_unemployment_rate",
                        "L1_inflation_hicp",
                        "L1_long_rate",
                        "L1_pop_growth",
                    ],
                    "required": [
                        "L1_gdp_pc_growth",
                        "L1_unemployment_rate",
                        "L1_inflation_hicp",
                        "L1_long_rate",
                        "L1_pop_growth",
                    ],
                },
            ),
            (
                "origin_income_comp",
                {
                    "terms": ["L1_origin_gdp_pc_ppp_const_wavg_log"],
                    "required": ["L1_origin_gdp_pc_ppp_const_wavg_log"],
                },
            ),
        ]
    )
    annual_headline_seq = ["migration", "air", "macros", "origin_income_comp"]
    annual_seq_df, annual_lobo_df, annual_shap_df, annual_meta = evaluate_block_decomposition(
        annual,
        sample_name="annual_headline_composition",
        y="hpi_growth",
        index_cols=("geo", "year"),
        blocks=annual_headline_blocks,
        sequence=annual_headline_seq,
        cluster_time=True,
        preprocessor=prep_annual_comp,
    )

    # Annual extended sample including region-share composition terms (Table 6 Columns 3-4 sample).
    annual_groups_blocks = OrderedDict(annual_headline_blocks)
    annual_groups_blocks["origin_region_shares"] = {
        "terms": ["L1_share_group_LATAM_CARIB", "L1_share_group_EU_EEA_CH_UK", "L1_share_group_MENA"],
        "required": ["L1_share_group_LATAM_CARIB", "L1_share_group_EU_EEA_CH_UK", "L1_share_group_MENA"],
    }
    annual_groups_seq = ["migration", "air", "macros", "origin_income_comp", "origin_region_shares"]
    annual_g_seq_df, annual_g_lobo_df, annual_g_shap_df, annual_g_meta = evaluate_block_decomposition(
        annual,
        sample_name="annual_extended_composition_groups",
        y="hpi_growth",
        index_cols=("geo", "year"),
        blocks=annual_groups_blocks,
        sequence=annual_groups_seq,
        cluster_time=True,
        preprocessor=prep_annual_comp,
    )

    # Quarterly traveler-quality full interacted sample.
    quarterly_blocks = OrderedDict(
        [
            ("air_growth", {"terms": ["L1_air_yoy"], "required": ["L1_air_yoy"]}),
            ("route_shocks", {"terms": ["L1_open_rate_norm_q", "L1_close_rate_norm_q"], "required": ["L1_open_rate_norm_q", "L1_close_rate_norm_q"]}),
            ("airfare_proxy", {"terms": ["L1_airfare_yoy_q_c"], "required": ["L1_airfare_yoy_q_c"]}),
            ("airline_mix_level", {"terms": ["L1_lic_neu_share_pas_c"], "required": ["L1_lic_neu_share_pas_c"]}),
            ("operating_intensity", {"terms": ["L1_pax_per_move_total"], "required": ["L1_pax_per_move_total"]}),
            (
                "interactions",
                {
                    "terms": ["L1_air_yoy:L1_lic_neu_share_pas_c", "L1_air_yoy:L1_airfare_yoy_q_c"],
                    "required": ["L1_air_yoy", "L1_lic_neu_share_pas_c", "L1_airfare_yoy_q_c"],
                },
            ),
        ]
    )
    quarterly_seq = ["air_growth", "route_shocks", "airfare_proxy", "airline_mix_level", "operating_intensity", "interactions"]
    q_seq_df, q_lobo_df, q_shap_df, q_meta = evaluate_block_decomposition(
        quarterly,
        sample_name="quarterly_traveler_quality_full_interact",
        y="hpi_yoy",
        index_cols=("geo", "quarter_id"),
        blocks=quarterly_blocks,
        sequence=quarterly_seq,
        cluster_time=True,
        preprocessor=prep_quarterly_tq,
    )

    seq_all = pd.concat([annual_seq_df, annual_g_seq_df, q_seq_df], ignore_index=True)
    lobo_all = pd.concat([annual_lobo_df, annual_g_lobo_df, q_lobo_df], ignore_index=True)
    shap_all = pd.concat([annual_shap_df, annual_g_shap_df, q_shap_df], ignore_index=True)

    seq_path = RESULTS_DIR / "model_fit_sequence.csv"
    lobo_path = RESULTS_DIR / "model_fit_leave_one_block_out.csv"
    shap_path = RESULTS_DIR / "model_fit_shapley.csv"
    seq_all.to_csv(seq_path, index=False)
    lobo_all.to_csv(lobo_path, index=False)
    shap_all.to_csv(shap_path, index=False)

    # Compact text summary for quick reading.
    summary_lines = []
    for meta in [annual_meta, annual_g_meta, q_meta]:
        s = meta["sample"]
        full = meta["full_model"]
        summary_lines.append(f"# {s}")
        summary_lines.append(f"Common sample rows={meta['rows_common_sample']}, entities={meta['entities_common_sample']}, periods={meta['periods_common_sample']}")
        summary_lines.append(
            f"Full model: R2_inclusive={full['r2_inclusive']:.4f}, R2_within={full['r2_within']:.4f}, "
            f"RMSE={full['rmse']:.4f}, MAE={full['mae']:.4f}, Corr(actual,fitted)={full['corr_actual_fitted']:.4f}"
        )
        summary_lines.append("Top leave-one-block-out losses (delta positive means block improves full model):")
        sub = lobo_all[lobo_all["sample"] == s].sort_values("delta_rmse", ascending=False)
        for _, r in sub.iterrows():
            summary_lines.append(
                f"  - {r['factor']}: +{r['delta_rmse']:.4f} RMSE, +{r['delta_mae']:.4f} MAE, "
                f"+{r['delta_r2_within']:.4f} within-R2, +{r['delta_r2_inclusive']:.4f} incl-R2"
            )
        summary_lines.append("Shapley contributions (within-R2):")
        ssub = shap_all[shap_all["sample"] == s].sort_values("shapley_r2_within", ascending=False)
        for _, r in ssub.iterrows():
            summary_lines.append(
                f"  - {r['factor']}: within={r['shapley_r2_within']:.4f}, inclusive={r['shapley_r2_inclusive']:.4f}"
            )
        summary_lines.append("")

    txt_path = RESULTS_DIR / "model_fit_summary.txt"
    txt_path.write_text("\n".join(summary_lines))

    meta_path = META_DIR / "model_fit_decomposition_summary.json"
    meta_obj = {"samples": [annual_meta, annual_g_meta, q_meta]}
    meta_path.write_text(json.dumps(meta_obj, indent=2))

    print(f"[ok] wrote {seq_path}")
    print(f"[ok] wrote {lobo_path}")
    print(f"[ok] wrote {shap_path}")
    print(f"[ok] wrote {txt_path}")
    print(f"[ok] wrote {meta_path}")


if __name__ == "__main__":
    main()
