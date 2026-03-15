#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
META_DIR = ROOT / "data" / "metadata"
PAPER_TABLES_DIR = ROOT / "paper_overleaf" / "tables"


def winsorize_by_train(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], p_low=0.01, p_high=0.99):
    train = train.copy()
    test = test.copy()
    for c in cols:
        if c not in train.columns:
            continue
        x = pd.to_numeric(train[c], errors="coerce")
        if x.notna().sum() == 0:
            continue
        lo, hi = x.quantile([p_low, p_high])
        train[c] = pd.to_numeric(train[c], errors="coerce").clip(lo, hi)
        if c in test.columns:
            test[c] = pd.to_numeric(test[c], errors="coerce").clip(lo, hi)
    return train, test


def build_design(train: pd.DataFrame, test: pd.DataFrame, y_col: str, x_cols: list[str], entity_col="geo"):
    train = train.dropna(subset=[y_col, entity_col] + x_cols).copy()
    test = test.dropna(subset=[y_col, entity_col] + x_cols).copy()
    if train.empty or test.empty:
        return None

    seen = set(train[entity_col].astype(str).unique())
    test = test[test[entity_col].astype(str).isin(seen)].copy()
    if test.empty:
        return None

    train_ent = pd.get_dummies(train[entity_col].astype(str), prefix="geo", drop_first=True)
    test_ent = pd.get_dummies(test[entity_col].astype(str), prefix="geo", drop_first=True)
    train_ent, test_ent = train_ent.align(test_ent, join="outer", axis=1, fill_value=0)

    X_train = pd.concat([train[x_cols].astype(float), train_ent.astype(float)], axis=1)
    X_test = pd.concat([test[x_cols].astype(float), test_ent.astype(float)], axis=1)
    X_train.insert(0, "const", 1.0)
    X_test.insert(0, "const", 1.0)

    y_train = train[y_col].astype(float).values
    return train, test, X_train, X_test, y_train


def ols_predict(train: pd.DataFrame, test: pd.DataFrame, y_col: str, x_cols: list[str], entity_col="geo"):
    built = build_design(train, test, y_col, x_cols, entity_col=entity_col)
    if built is None:
        return pd.DataFrame()
    train2, test2, X_train, X_test, y_train = built

    beta, *_ = np.linalg.lstsq(X_train.values, y_train, rcond=None)
    out = test2.copy()
    out["yhat"] = X_test.values @ beta

    built0 = build_design(train, test, y_col, [], entity_col=entity_col)
    if built0 is not None:
        _, test0, X0_train, X0_test, y0_train = built0
        beta0, *_ = np.linalg.lstsq(X0_train.values, y0_train, rcond=None)
        key_cols = [entity_col]
        if "year" in out.columns:
            key_cols.append("year")
        if "quarter_id" in out.columns:
            key_cols.append("quarter_id")
        b = test0[key_cols].copy()
        b["yhat_fe_only"] = X0_test.values @ beta0
        out = out.merge(b, on=key_cols, how="left")
    else:
        out["yhat_fe_only"] = np.nan
    return out


def pooled_metrics(df: pd.DataFrame, y_col="y") -> dict:
    x = pd.to_numeric(df[y_col], errors="coerce")
    p = pd.to_numeric(df["yhat"], errors="coerce")
    e = x - p
    out = {
        "nobs_test": int(df[[y_col, "yhat"]].dropna().shape[0]),
        "rmse": float(np.sqrt(np.nanmean(e**2))),
        "mae": float(np.nanmean(np.abs(e))),
        "corr_actual_pred": float(pd.concat([x, p], axis=1).corr().iloc[0, 1]),
    }
    if "yhat_fe_only" in df.columns:
        p0 = pd.to_numeric(df["yhat_fe_only"], errors="coerce")
        mask = x.notna() & p.notna() & p0.notna()
        if mask.any():
            sse = float(np.sum((x[mask] - p[mask]) ** 2))
            sse0 = float(np.sum((x[mask] - p0[mask]) ** 2))
            out["oos_r2_vs_fe_only"] = float(1.0 - sse / sse0) if sse0 > 0 else np.nan
        else:
            out["oos_r2_vs_fe_only"] = np.nan
    return out


def rank_metrics(df: pd.DataFrame, period_col: str, y_col="y") -> dict:
    rows = []
    for p, g in df.groupby(period_col):
        g = g.dropna(subset=["yhat", y_col]).copy()
        if len(g) < 5:
            continue
        ic = spearmanr(g["yhat"], g[y_col], nan_policy="omit").correlation
        g = g.sort_values("yhat")
        k = max(1, len(g) // 4)
        spread = float(g.iloc[-k:][y_col].mean() - g.iloc[:k][y_col].mean())
        rows.append({"period": p, "rank_ic": ic, "top25_minus_bottom25": spread})
    if not rows:
        return {"periods_eval": 0, "mean_rank_ic": np.nan, "mean_top25_minus_bottom25": np.nan}
    d = pd.DataFrame(rows)
    return {
        "periods_eval": int(len(d)),
        "mean_rank_ic": float(d["rank_ic"].mean(skipna=True)),
        "mean_top25_minus_bottom25": float(d["top25_minus_bottom25"].mean(skipna=True)),
        "median_top25_minus_bottom25": float(d["top25_minus_bottom25"].median(skipna=True)),
    }


def expanding_backtest(df: pd.DataFrame, *, sample: str, y_col: str, period_col: str, x_cols: list[str], min_train: int) -> tuple[pd.DataFrame, dict]:
    d = df.replace([np.inf, -np.inf], np.nan).copy()
    periods = sorted(pd.to_numeric(d[period_col], errors="coerce").dropna().astype(int).unique().tolist())
    preds = []
    for i, p in enumerate(periods):
        if i < min_train:
            continue
        train = d[pd.to_numeric(d[period_col], errors="coerce").astype("Int64") < p].copy()
        test = d[pd.to_numeric(d[period_col], errors="coerce").astype("Int64") == p].copy()
        if train.empty or test.empty:
            continue
        train, test = winsorize_by_train(train, test, x_cols)
        out = ols_predict(train, test, y_col=y_col, x_cols=x_cols, entity_col="geo")
        if out.empty:
            continue
        out["sample"] = sample
        out["forecast_period"] = p
        out["y"] = pd.to_numeric(out[y_col], errors="coerce")
        preds.append(out)

    if not preds:
        return pd.DataFrame(), {"sample": sample}
    pred_df = pd.concat(preds, ignore_index=True)
    m = {"sample": sample, **pooled_metrics(pred_df, y_col="y"), **rank_metrics(pred_df, "forecast_period", y_col="y")}
    return pred_df, m


def add_release_lags(annual: pd.DataFrame, quarterly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = annual.sort_values(["geo", "year"]).copy()
    g = a.groupby("geo", sort=False)
    for col in [
        "hpi_growth",
        "net_migration_rate_countryweb_patch",
        "asylum_rate_per_1000",
        "non_asylum_immigration_rate_per_1000",
        "naturalization_rate_per_1000",
        "acq_origin_gdp_wavg_log",
        "car_reg_total_per_1000",
        "car_reg_elc_share",
        "air_growth",
        "gdp_pc_growth",
        "unemployment_rate",
        "inflation_hicp",
        "long_rate",
        "pop_growth",
    ]:
        if col in a.columns:
            a[f"L1_{col}"] = g[col].shift(1)
            a[f"L2_{col}"] = g[col].shift(2)

    q = quarterly.sort_values(["geo", "quarter_id"]).copy()
    gq = q.groupby("geo", sort=False)
    for col in ["hpi_yoy", "air_yoy", "tour_total_yoy", "asylum_first_q_per100k", "tps_ua_stock_qe_per100k", "pax_per_move_total", "airfare_yoy_q"]:
        if col in q.columns:
            q[f"L1_{col}"] = gq[col].shift(1)
            q[f"L2_{col}"] = gq[col].shift(2)
    return a, q


def write_table(metrics: pd.DataFrame) -> None:
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    def row(sample):
        r = metrics[metrics["sample"] == sample]
        if r.empty:
            return ["", "", "", "", "", ""]
        x = r.iloc[0]
        return [
            f"{x.get('nobs_test', np.nan):.0f}",
            f"{x.get('rmse', np.nan):.2f}",
            f"{x.get('mae', np.nan):.2f}",
            f"{x.get('corr_actual_pred', np.nan):.2f}",
            f"{x.get('oos_r2_vs_fe_only', np.nan):.2f}",
            f"{x.get('mean_top25_minus_bottom25', np.nan):.2f}",
        ]

    rows = {
        "Annual naive timing": row("annual_naive"),
        "Annual release-aware": row("annual_release_aware"),
        "Quarterly naive timing": row("quarterly_naive"),
        "Quarterly release-aware": row("quarterly_release_aware"),
    }

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Predictive performance with and without release-aware information timing}",
        r"\label{tab:release_aware_predictive}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{5.2cm}rrrrrr}",
        r"\toprule",
        r"Backtest setup & Obs. & RMSE & MAE & Corr(actual,pred) & OOS $R^2$ vs FE-only & Mean top25-bottom25 (pp) \\",
        r"\midrule",
    ]
    for name, vals in rows.items():
        lines.append(name + " & " + " & ".join(vals) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Notes: Release-aware models impose additional information delays by source before forecasting: annual migration/composition/citizenship/car variables are shifted by one extra year beyond standard lagging; quarterly air/tourism/airfare/operating-intensity variables are shifted by one extra quarter, while asylum and temporary-protection indicators keep a shorter delay. This approximates real-time information availability rather than ex-post explanatory fit.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    (PAPER_TABLES_DIR / "tab_release_aware_predictive.tex").write_text("\n".join(lines))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    annual = pd.read_parquet(PROC_DIR / "panel_annual_extended_channels.parquet")
    # Harmonized-first canonical annual series.
    annual["hpi_growth"] = pd.to_numeric(annual.get("hpi_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(annual.get("hpi_growth"), errors="coerce")
    )
    annual["air_growth"] = pd.to_numeric(annual.get("air_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(annual.get("air_growth"), errors="coerce")
    )
    annual["gdp_pc_growth"] = pd.to_numeric(annual.get("gdp_pc_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(annual.get("gdp_pc_growth"), errors="coerce")
    )
    annual["unemployment_rate"] = pd.to_numeric(annual.get("unemployment_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(annual.get("unemployment_rate"), errors="coerce")
    )
    annual["inflation_hicp"] = pd.to_numeric(annual.get("inflation_hicp_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(annual.get("inflation_hicp"), errors="coerce")
    )
    annual["long_rate"] = pd.to_numeric(annual.get("long_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(annual.get("long_rate"), errors="coerce")
    )
    annual["pop_growth"] = pd.to_numeric(annual.get("pop_growth_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(annual.get("pop_growth"), errors="coerce")
    )
    annual["net_migration_rate"] = pd.to_numeric(annual.get("net_migration_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(annual.get("net_migration_rate"), errors="coerce")
    )

    # Pull canonical blended migration series from harmonized core stream.
    core_h_path = PROC_DIR / "panel_annual_harmonized.parquet"
    if not core_h_path.exists():
        raise FileNotFoundError(
            "Missing `panel_annual_harmonized.parquet`. Run scripts/harmonize_cross_frequency.py first."
        )
    core = pd.read_parquet(core_h_path)
    core["net_migration_rate_core"] = pd.to_numeric(core.get("net_migration_rate_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(core.get("net_migration_rate"), errors="coerce")
    )
    core_patch = core[["geo", "year", "net_migration_rate_core"]].drop_duplicates(["geo", "year"]).rename(
        columns={"net_migration_rate_core": "net_migration_rate_countryweb_patch_core"}
    )
    annual = annual.merge(core_patch, on=["geo", "year"], how="left")

    if "net_migration_rate_countryweb_patch" not in annual.columns:
        annual["net_migration_rate_countryweb_patch"] = np.nan
    annual["net_migration_rate_countryweb_patch"] = annual["net_migration_rate_countryweb_patch"].combine_first(
        annual["net_migration_rate_countryweb_patch_core"]
    )

    overlay_path = PROC_DIR / "panel_annual_countryweb_overlay.parquet"
    if overlay_path.exists():
        overlay = pd.read_parquet(overlay_path)[["geo", "year", "net_migration_rate_countryweb_patch"]].drop_duplicates(["geo", "year"])
        annual = annual.merge(
            overlay.rename(columns={"net_migration_rate_countryweb_patch": "net_migration_rate_countryweb_patch_overlay"}),
            on=["geo", "year"],
            how="left",
        )
        annual["net_migration_rate_countryweb_patch"] = annual["net_migration_rate_countryweb_patch"].combine_first(
            annual["net_migration_rate_countryweb_patch_overlay"]
        )

    annual["net_migration_rate_countryweb_patch"] = annual["net_migration_rate_countryweb_patch"].fillna(annual["net_migration_rate"])
    quarterly = pd.read_parquet(PROC_DIR / "panel_quarterly_extended_channels.parquet")
    quarterly["hpi_yoy"] = pd.to_numeric(quarterly.get("hpi_yoy_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(quarterly.get("hpi_yoy"), errors="coerce")
    )
    if "quarter_id" not in quarterly.columns:
        p = pd.PeriodIndex(quarterly["period_str"], freq="Q")
        quarterly["quarter_id"] = p.year * 4 + p.quarter

    annual, quarterly = add_release_lags(annual, quarterly)

    annual_naive_cols = [
        "L1_net_migration_rate_countryweb_patch",
        "L1_asylum_rate_per_1000",
        "L1_non_asylum_immigration_rate_per_1000",
        "L1_naturalization_rate_per_1000",
        "L1_acq_origin_gdp_wavg_log",
        "L1_car_reg_total_per_1000",
        "L1_car_reg_elc_share",
        "L1_air_growth",
        "L1_gdp_pc_growth",
        "L1_unemployment_rate",
        "L1_inflation_hicp",
        "L1_long_rate",
        "L1_pop_growth",
    ]
    annual_release_cols = [
        "L2_net_migration_rate_countryweb_patch",
        "L2_asylum_rate_per_1000",
        "L2_non_asylum_immigration_rate_per_1000",
        "L2_naturalization_rate_per_1000",
        "L2_acq_origin_gdp_wavg_log",
        "L2_car_reg_total_per_1000",
        "L2_car_reg_elc_share",
        "L1_air_growth",
        "L1_gdp_pc_growth",
        "L1_unemployment_rate",
        "L1_inflation_hicp",
        "L1_long_rate",
        "L1_pop_growth",
    ]

    quarterly_naive_cols = [
        "L1_air_yoy",
        "L1_tour_total_yoy",
        "L1_asylum_first_q_per100k",
        "L1_tps_ua_stock_qe_per100k",
        "L1_pax_per_move_total",
        "L1_airfare_yoy_q",
    ]
    quarterly_release_cols = [
        "L2_air_yoy",
        "L2_tour_total_yoy",
        "L1_asylum_first_q_per100k",
        "L1_tps_ua_stock_qe_per100k",
        "L2_pax_per_move_total",
        "L2_airfare_yoy_q",
    ]

    a_naive_pred, a_naive_m = expanding_backtest(
        annual,
        sample="annual_naive",
        y_col="hpi_growth",
        period_col="year",
        x_cols=annual_naive_cols,
        min_train=4,
    )
    a_rel_pred, a_rel_m = expanding_backtest(
        annual,
        sample="annual_release_aware",
        y_col="hpi_growth",
        period_col="year",
        x_cols=annual_release_cols,
        min_train=4,
    )

    q_naive_pred, q_naive_m = expanding_backtest(
        quarterly,
        sample="quarterly_naive",
        y_col="hpi_yoy",
        period_col="quarter_id",
        x_cols=quarterly_naive_cols,
        min_train=20,
    )
    q_rel_pred, q_rel_m = expanding_backtest(
        quarterly,
        sample="quarterly_release_aware",
        y_col="hpi_yoy",
        period_col="quarter_id",
        x_cols=quarterly_release_cols,
        min_train=20,
    )

    preds = pd.concat([a_naive_pred, a_rel_pred, q_naive_pred, q_rel_pred], ignore_index=True, sort=False)
    metrics = pd.DataFrame([a_naive_m, a_rel_m, q_naive_m, q_rel_m])

    preds.to_csv(RESULTS_DIR / "release_aware_predictions.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "release_aware_metrics.csv", index=False)
    write_table(metrics)

    meta = {
        "annual_naive": a_naive_m,
        "annual_release_aware": a_rel_m,
        "quarterly_naive": q_naive_m,
        "quarterly_release_aware": q_rel_m,
    }
    (META_DIR / "release_aware_summary.json").write_text(json.dumps(meta, indent=2))

    print(f"[ok] wrote {RESULTS_DIR / 'release_aware_metrics.csv'}")
    print(f"[ok] wrote {PAPER_TABLES_DIR / 'tab_release_aware_predictive.tex'}")


if __name__ == "__main__":
    main()
