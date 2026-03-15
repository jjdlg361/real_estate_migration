#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
META_DIR = ROOT / "data" / "metadata"
PAPER_FIGS_DIR = ROOT / "paper_overleaf" / "figures"


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


def add_quarterly_interactions(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()
    mu_airfare = pd.to_numeric(train["L1_airfare_yoy_q"], errors="coerce").mean(skipna=True)
    mu_lic = pd.to_numeric(train["L1_lic_neu_share_pas"], errors="coerce").mean(skipna=True)
    for d in [train, test]:
        d["L1_airfare_yoy_q_c_bt"] = pd.to_numeric(d["L1_airfare_yoy_q"], errors="coerce") - mu_airfare
        d["L1_lic_neu_share_pas_c_bt"] = pd.to_numeric(d["L1_lic_neu_share_pas"], errors="coerce") - mu_lic
        d["L1_air_yoy_x_airfare_bt"] = pd.to_numeric(d["L1_air_yoy"], errors="coerce") * d["L1_airfare_yoy_q_c_bt"]
        d["L1_air_yoy_x_lic_bt"] = pd.to_numeric(d["L1_air_yoy"], errors="coerce") * d["L1_lic_neu_share_pas_c_bt"]
    return train, test


def build_design(train: pd.DataFrame, test: pd.DataFrame, y_col: str, x_cols: list[str], entity_col="geo"):
    # Keep only rows with required values.
    train = train.dropna(subset=[y_col, entity_col] + x_cols).copy()
    test = test.dropna(subset=[y_col, entity_col] + x_cols).copy()
    if train.empty or test.empty:
        return None

    # Restrict test to entities seen in train (investor-style expanding window for known markets).
    seen = set(train[entity_col].astype(str).unique())
    test = test[test[entity_col].astype(str).isin(seen)].copy()
    if test.empty:
        return None

    # One-hot entity FE (drop first to avoid collinearity with intercept).
    train_ent = pd.get_dummies(train[entity_col].astype(str), prefix="geo", drop_first=True)
    test_ent = pd.get_dummies(test[entity_col].astype(str), prefix="geo", drop_first=True)
    train_ent, test_ent = train_ent.align(test_ent, join="outer", axis=1, fill_value=0)

    X_train = pd.concat([train[x_cols].astype(float), train_ent.astype(float)], axis=1)
    X_test = pd.concat([test[x_cols].astype(float), test_ent.astype(float)], axis=1)
    X_train.insert(0, "const", 1.0)
    X_test.insert(0, "const", 1.0)

    y_train = train[y_col].astype(float).values
    y_test = test[y_col].astype(float).values
    return train, test, X_train, X_test, y_train, y_test


def ols_predict(train: pd.DataFrame, test: pd.DataFrame, y_col: str, x_cols: list[str], entity_col="geo"):
    built = build_design(train, test, y_col, x_cols, entity_col=entity_col)
    if built is None:
        return pd.DataFrame()
    train2, test2, X_train, X_test, y_train, _ = built

    beta, *_ = np.linalg.lstsq(X_train.values, y_train, rcond=None)
    yhat = X_test.values @ beta

    out = test2.copy()
    out["yhat"] = yhat

    # FE-only benchmark (entity dummies only)
    built0 = build_design(train, test, y_col, [], entity_col=entity_col)
    if built0 is not None:
        _, test0, X0_train, X0_test, y0_train, _ = built0
        beta0, *_ = np.linalg.lstsq(X0_train.values, y0_train, rcond=None)
        yhat0 = X0_test.values @ beta0
        key_cols = [entity_col]
        if "year" in out.columns:
            key_cols.append("year")
        if "quarter_id" in out.columns:
            key_cols.append("quarter_id")
        bmark = test0[key_cols].copy()
        bmark["yhat_fe_only"] = yhat0
        out = out.merge(bmark, on=key_cols, how="left")
    else:
        out["yhat_fe_only"] = np.nan
    return out


def pooled_metrics(df: pd.DataFrame, y_col: str = "y") -> dict[str, float]:
    x = pd.to_numeric(df[y_col], errors="coerce")
    p = pd.to_numeric(df["yhat"], errors="coerce")
    e = x - p
    metrics = {
        "nobs_test": int(df[[y_col, "yhat"]].dropna().shape[0]),
        "rmse": float(np.sqrt(np.nanmean(e**2))),
        "mae": float(np.nanmean(np.abs(e))),
        "corr_actual_pred": float(pd.concat([x, p], axis=1).corr().iloc[0, 1]),
    }
    if "yhat_fe_only" in df.columns:
        p0 = pd.to_numeric(df["yhat_fe_only"], errors="coerce")
        e0 = x - p0
        mask = x.notna() & p.notna() & p0.notna()
        if mask.any():
            sse = float(np.sum((x[mask] - p[mask]) ** 2))
            sse0 = float(np.sum((x[mask] - p0[mask]) ** 2))
            metrics["rmse_fe_only"] = float(np.sqrt(np.mean((x[mask] - p0[mask]) ** 2)))
            metrics["mae_fe_only"] = float(np.mean(np.abs(x[mask] - p0[mask])))
            metrics["oos_r2_vs_fe_only"] = float(1.0 - sse / sse0) if sse0 > 0 else np.nan
        else:
            metrics["rmse_fe_only"] = np.nan
            metrics["mae_fe_only"] = np.nan
            metrics["oos_r2_vs_fe_only"] = np.nan
    return metrics


def cross_sectional_investor_metrics(df: pd.DataFrame, period_col: str, y_col: str = "y") -> dict[str, float]:
    rows = []
    for p, g in df.groupby(period_col):
        g = g.dropna(subset=[y_col, "yhat"]).copy()
        if len(g) < 5:
            continue
        # Rank IC (cross-sectional)
        ic = spearmanr(g["yhat"], g[y_col], nan_policy="omit").correlation
        # Tercile spread (top predicted - bottom predicted)
        g = g.sort_values("yhat")
        n = len(g)
        k = max(1, n // 3)
        bottom = g.iloc[:k][y_col].mean()
        top = g.iloc[-k:][y_col].mean()
        rows.append({"period": p, "rank_ic": ic, "top_minus_bottom": float(top - bottom), "n": int(n)})
    if not rows:
        return {"periods_eval": 0, "mean_rank_ic": np.nan, "mean_top_minus_bottom": np.nan}
    d = pd.DataFrame(rows)
    return {
        "periods_eval": int(len(d)),
        "mean_rank_ic": float(d["rank_ic"].mean(skipna=True)),
        "median_rank_ic": float(d["rank_ic"].median(skipna=True)),
        "mean_top_minus_bottom": float(d["top_minus_bottom"].mean(skipna=True)),
        "median_top_minus_bottom": float(d["top_minus_bottom"].median(skipna=True)),
    }


def expanding_backtest(
    df: pd.DataFrame,
    *,
    sample_name: str,
    y_col: str,
    period_col: str,
    entity_col: str,
    x_cols: list[str],
    min_train_periods: int,
    winsor_cols: list[str] | None = None,
    preprocess_fold=None,
) -> tuple[pd.DataFrame, dict]:
    d = df.replace([np.inf, -np.inf], np.nan).copy()
    periods = sorted(pd.to_numeric(d[period_col], errors="coerce").dropna().astype(int).unique().tolist())
    preds = []
    for i, p in enumerate(periods):
        if i < min_train_periods:
            continue
        train = d[pd.to_numeric(d[period_col], errors="coerce").astype("Int64") < p].copy()
        test = d[pd.to_numeric(d[period_col], errors="coerce").astype("Int64") == p].copy()
        if train.empty or test.empty:
            continue
        if preprocess_fold is not None:
            train, test = preprocess_fold(train, test)
        if winsor_cols:
            train, test = winsorize_by_train(train, test, winsor_cols)
        out = ols_predict(train, test, y_col, x_cols, entity_col=entity_col)
        if out.empty:
            continue
        out = out.copy()
        out["sample"] = sample_name
        out["forecast_period"] = p
        out["y"] = pd.to_numeric(out[y_col], errors="coerce")
        preds.append(out)
    if not preds:
        return pd.DataFrame(), {"sample": sample_name}
    pred_df = pd.concat(preds, ignore_index=True)
    base = pooled_metrics(pred_df, y_col="y")
    investor = cross_sectional_investor_metrics(pred_df, "forecast_period", y_col="y")
    meta = {"sample": sample_name, **base, **investor}
    return pred_df, meta


def prep_annual(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def prep_quarterly_fold(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()
    return add_quarterly_interactions(train, test)


def plot_scatter(preds_all: pd.DataFrame, metrics_df: pd.DataFrame):
    PAPER_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    samples = [
        ("annual_investor_headline", "Annual investor-style backtest"),
        ("quarterly_investor_tq_full", "Quarterly investor-style backtest"),
    ]
    colors = {"annual_investor_headline": "#1f4e79", "quarterly_investor_tq_full": "#b22222"}
    for ax, (s, title) in zip(axes, samples):
        d = preds_all[preds_all["sample"] == s].dropna(subset=["y", "yhat"]).copy()
        if d.empty:
            ax.text(0.5, 0.5, "No predictions", ha="center", va="center", transform=ax.transAxes)
            continue
        x = d["y"].astype(float)
        y = d["yhat"].astype(float)
        ax.scatter(x, y, s=18, alpha=0.45, color=colors[s], edgecolor="none")
        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1.2, color="#333333")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Actual next-period house-price growth")
        ax.set_ylabel("Predicted next-period growth")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        m = metrics_df[metrics_df["sample"] == s]
        if not m.empty:
            r = m.iloc[0]
            txt = (
                f"RMSE={r['rmse']:.2f}\n"
                f"MAE={r['mae']:.2f}\n"
                f"Corr={r['corr_actual_pred']:.2f}\n"
                f"OOS $R^2$ vs FE-only={r.get('oos_r2_vs_fe_only', np.nan):.2f}"
            )
            ax.text(
                0.03, 0.97, txt,
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="#bbbbbb", alpha=0.9, boxstyle="round,pad=0.35")
            )
    fig.suptitle("Predicted vs actual next-period house-price growth (expanding-window, investor-style backtest)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(PAPER_FIGS_DIR / "fig_investor_style_pred_vs_actual.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS_DIR.mkdir(parents=True, exist_ok=True)

    annual = pd.read_parquet(PROC_DIR / "panel_annual_migration_composition.parquet")
    quarterly = pd.read_parquet(PROC_DIR / "panel_quarterly_traveler_quality.parquet")

    # Annual headline sample aligned with the paper's composition headline model.
    annual = prep_annual(annual)
    annual_preds, annual_meta = expanding_backtest(
        annual,
        sample_name="annual_investor_headline",
        y_col="hpi_growth",
        period_col="year",
        entity_col="geo",
        x_cols=[
            "L1_net_migration_rate",
            "L1_air_growth",
            "L1_gdp_pc_growth",
            "L1_unemployment_rate",
            "L1_inflation_hicp",
            "L1_long_rate",
            "L1_pop_growth",
            "L1_origin_gdp_pc_ppp_const_wavg_log",
        ],
        min_train_periods=4,
        winsor_cols=[
            "L1_net_migration_rate",
            "L1_air_growth",
            "L1_gdp_pc_growth",
            "L1_unemployment_rate",
            "L1_inflation_hicp",
            "L1_long_rate",
            "L1_pop_growth",
            "L1_origin_gdp_pc_ppp_const_wavg_log",
        ],
    )

    # Quarterly traveler-quality spec (no future quarter FE in backtest).
    quarterly_preds, quarterly_meta = expanding_backtest(
        quarterly,
        sample_name="quarterly_investor_tq_full",
        y_col="hpi_yoy",
        period_col="quarter_id",
        entity_col="geo",
        x_cols=[
            "L1_air_yoy",
            "L1_open_rate_norm_q",
            "L1_close_rate_norm_q",
            "L1_airfare_yoy_q_c_bt",
            "L1_lic_neu_share_pas_c_bt",
            "L1_pax_per_move_total",
            "L1_air_yoy_x_airfare_bt",
            "L1_air_yoy_x_lic_bt",
        ],
        min_train_periods=20,
        winsor_cols=[
            "L1_air_yoy",
            "L1_open_rate_norm_q",
            "L1_close_rate_norm_q",
            "L1_airfare_yoy_q",
            "L1_lic_neu_share_pas",
            "L1_pax_per_move_total",
        ],
        preprocess_fold=prep_quarterly_fold,
    )

    preds_all = pd.concat([annual_preds, quarterly_preds], ignore_index=True, sort=False)
    preds_path = RESULTS_DIR / "investor_style_forecast_predictions.csv"
    preds_all.to_csv(preds_path, index=False)

    metrics_df = pd.DataFrame([annual_meta, quarterly_meta])
    metrics_path = RESULTS_DIR / "investor_style_forecast_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Period-level metrics for transparency
    period_rows = []
    for sample_name, d in preds_all.groupby("sample"):
        pcol = "forecast_period"
        for p, g in d.groupby(pcol):
            g = g.dropna(subset=["y", "yhat"])
            if g.empty:
                continue
            err = g["y"] - g["yhat"]
            row = {
                "sample": sample_name,
                "forecast_period": int(p),
                "nobs": int(len(g)),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
                "corr_actual_pred": float(pd.concat([g["y"], g["yhat"]], axis=1).corr().iloc[0, 1]) if len(g) > 1 else np.nan,
            }
            if len(g) >= 5:
                ic = spearmanr(g["yhat"], g["y"], nan_policy="omit").correlation
                g2 = g.sort_values("yhat")
                k = max(1, len(g2)//3)
                row["rank_ic"] = float(ic) if ic is not None else np.nan
                row["top_minus_bottom"] = float(g2.iloc[-k:]["y"].mean() - g2.iloc[:k]["y"].mean())
            period_rows.append(row)
    period_df = pd.DataFrame(period_rows)
    period_path = RESULTS_DIR / "investor_style_forecast_period_metrics.csv"
    period_df.to_csv(period_path, index=False)

    plot_scatter(preds_all, metrics_df)

    meta = {
        "annual_investor_headline": annual_meta,
        "quarterly_investor_tq_full": quarterly_meta,
    }
    (META_DIR / "investor_style_forecast_summary.json").write_text(json.dumps(meta, indent=2))

    print(f"[ok] wrote {preds_path}")
    print(f"[ok] wrote {metrics_path}")
    print(f"[ok] wrote {period_path}")
    print(f"[ok] wrote {PAPER_FIGS_DIR / 'fig_investor_style_pred_vs_actual.pdf'}")


if __name__ == "__main__":
    main()
