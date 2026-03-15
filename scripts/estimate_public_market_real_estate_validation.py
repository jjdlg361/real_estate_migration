#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
META_DIR = ROOT / "data" / "metadata"
PAPER_TABLES_DIR = ROOT / "paper_overleaf" / "tables"
PAPER_FIGS_DIR = ROOT / "paper_overleaf" / "figures"

UNIVERSE_CSV = RAW_DIR / "public_real_estate_ticker_universe.csv"
PRICES_CACHE = RAW_DIR / "public_real_estate_daily_prices.parquet"
TAIL_FRAC = 0.25
TAIL_PCT = int(round(TAIL_FRAC * 100))
TAIL_PAIR_LABEL = f"{TAIL_PCT}--{TAIL_PCT}"


@dataclass
class EvalOutput:
    merged: pd.DataFrame
    metrics: dict
    period_metrics: pd.DataFrame
    ls_period: pd.DataFrame


def _safe_mean_corr(x: pd.Series, y: pd.Series) -> float:
    z = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(z) < 2:
        return np.nan
    return float(z.corr().iloc[0, 1])


def strategy_perf_from_period_spreads(period_df: pd.DataFrame, *, spread_col: str = "top_minus_bottom", periods_per_year: int = 1) -> dict[str, float]:
    if period_df is None or period_df.empty or spread_col not in period_df.columns:
        return {
            "annualized_return_pct": np.nan,
            "annualized_vol_pct": np.nan,
            "sharpe_0rf": np.nan,
            "sortino_0rf": np.nan,
        }
    r = pd.to_numeric(period_df[spread_col], errors="coerce").dropna() / 100.0
    if r.empty:
        return {
            "annualized_return_pct": np.nan,
            "annualized_vol_pct": np.nan,
            "sharpe_0rf": np.nan,
            "sortino_0rf": np.nan,
        }
    ann_mean = float(r.mean() * periods_per_year)
    sd = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    ann_vol = float(sd * np.sqrt(periods_per_year)) if pd.notna(sd) else np.nan
    downside = r[r < 0]
    # Sortino is unstable with very few downside observations in short samples.
    down_sd = float(downside.std(ddof=1)) if len(downside) >= 3 else np.nan
    ann_down = float(down_sd * np.sqrt(periods_per_year)) if pd.notna(down_sd) else np.nan
    sharpe = float(ann_mean / ann_vol) if pd.notna(ann_vol) and ann_vol > 0 else np.nan
    sortino = float(ann_mean / ann_down) if pd.notna(ann_down) and ann_down > 0 else np.nan
    return {
        "annualized_return_pct": ann_mean * 100.0,
        "annualized_vol_pct": ann_vol * 100.0 if pd.notna(ann_vol) else np.nan,
        "sharpe_0rf": sharpe,
        "sortino_0rf": sortino,
    }


def strategy_perf_from_daily_returns(daily_df: pd.DataFrame, *, ret_col: str, trading_days_per_year: int = 252) -> dict[str, float]:
    if daily_df is None or daily_df.empty or ret_col not in daily_df.columns:
        return {
            "ann_return_pct": np.nan,
            "ann_vol_pct": np.nan,
            "sharpe_0rf": np.nan,
            "sortino_0rf": np.nan,
            "n_days": 0,
            "sample_years": np.nan,
        }
    r = pd.to_numeric(daily_df[ret_col], errors="coerce").dropna()
    if r.empty:
        return {
            "ann_return_pct": np.nan,
            "ann_vol_pct": np.nan,
            "sharpe_0rf": np.nan,
            "sortino_0rf": np.nan,
            "n_days": 0,
            "sample_years": np.nan,
        }
    mean_d = float(r.mean())
    sd_d = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    ann_ret = mean_d * trading_days_per_year
    ann_vol = sd_d * np.sqrt(trading_days_per_year) if pd.notna(sd_d) else np.nan
    downside = r[r < 0]
    down_sd = float(downside.std(ddof=1)) if len(downside) >= 20 else np.nan
    ann_down = down_sd * np.sqrt(trading_days_per_year) if pd.notna(down_sd) else np.nan
    sharpe = ann_ret / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan
    sortino = ann_ret / ann_down if pd.notna(ann_down) and ann_down > 0 else np.nan
    dts = pd.to_datetime(daily_df.get("date"), errors="coerce")
    years = (dts.max() - dts.min()).days / 365.25 if dts.notna().any() and dts.max() > dts.min() else np.nan
    return {
        "ann_return_pct": ann_ret * 100.0,
        "ann_vol_pct": ann_vol * 100.0 if pd.notna(ann_vol) else np.nan,
        "sharpe_0rf": float(sharpe) if pd.notna(sharpe) else np.nan,
        "sortino_0rf": float(sortino) if pd.notna(sortino) else np.nan,
        "n_days": int(len(r)),
        "sample_years": float(years) if pd.notna(years) else np.nan,
    }


def compute_tail_assignments(signal_df: pd.DataFrame, *, period_col: str, signal_col: str, tail_frac: float = TAIL_FRAC) -> pd.DataFrame:
    rows = []
    for p, g in signal_df.groupby(period_col):
        g = g.dropna(subset=[signal_col, "geo"]).copy()
        if g.empty:
            continue
        g["geo"] = g["geo"].astype(str).str.upper()
        g = g.sort_values(signal_col)
        n = len(g)
        k = max(1, int(np.ceil(n * tail_frac)))
        k = min(k, max(1, n // 2))
        g["rank_pos"] = np.arange(1, n + 1)
        g["is_bottom"] = g["rank_pos"] <= k
        g["is_top"] = g["rank_pos"] > (n - k)
        g["k_tail"] = int(k)
        g["tail_frac_realized"] = float(k / n) if n else np.nan
        g["period_n"] = int(n)
        rows.append(g[[period_col, "geo", signal_col, "is_top", "is_bottom", "k_tail", "tail_frac_realized", "period_n"]])
    if not rows:
        return pd.DataFrame(columns=[period_col, "geo", signal_col, "is_top", "is_bottom", "k_tail", "tail_frac_realized", "period_n"])
    return pd.concat(rows, ignore_index=True)


def load_universe() -> pd.DataFrame:
    u = pd.read_csv(UNIVERSE_CSV)
    u["country"] = u["country"].astype(str).str.upper()
    u["ticker"] = u["ticker"].astype(str).str.strip()
    u = u.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return u


def fetch_daily_prices(universe: pd.DataFrame, start: str = "2000-01-01") -> pd.DataFrame:
    tickers = sorted(universe["ticker"].unique().tolist())
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])

    raw = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="ticker",
    )
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])

    # yfinance can return either single-index or multi-index columns depending on ticker count.
    if isinstance(raw.columns, pd.MultiIndex):
        pieces = []
        level0 = list(raw.columns.get_level_values(0).unique())
        level1 = list(raw.columns.get_level_values(1).unique())
        if "Adj Close" in level0:
            adj = raw["Adj Close"].copy()
            for t in adj.columns:
                s = pd.to_numeric(adj[t], errors="coerce").dropna()
                if s.empty:
                    continue
                pieces.append(pd.DataFrame({"date": s.index, "ticker": t, "adj_close": s.values}))
        elif "Adj Close" in level1:
            for t in tickers:
                if (t, "Adj Close") in raw.columns:
                    s = pd.to_numeric(raw[(t, "Adj Close")], errors="coerce").dropna()
                    if s.empty:
                        continue
                    pieces.append(pd.DataFrame({"date": s.index, "ticker": t, "adj_close": s.values}))
        else:
            # Fallback to Close if adjusted closes are unavailable.
            if "Close" in level0:
                close = raw["Close"].copy()
                for t in close.columns:
                    s = pd.to_numeric(close[t], errors="coerce").dropna()
                    if s.empty:
                        continue
                    pieces.append(pd.DataFrame({"date": s.index, "ticker": t, "adj_close": s.values}))
            else:
                for t in tickers:
                    if (t, "Close") in raw.columns:
                        s = pd.to_numeric(raw[(t, "Close")], errors="coerce").dropna()
                        if s.empty:
                            continue
                        pieces.append(pd.DataFrame({"date": s.index, "ticker": t, "adj_close": s.values}))
        out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["date", "ticker", "adj_close"])
    else:
        c = "Adj Close" if "Adj Close" in raw.columns else "Close"
        s = pd.to_numeric(raw[c], errors="coerce").dropna()
        ticker = tickers[0]
        out = pd.DataFrame({"date": s.index, "ticker": ticker, "adj_close": s.values})

    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["ticker"] = out["ticker"].astype(str)
    out = out.dropna(subset=["adj_close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def build_country_daily_returns(prices: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = prices.merge(universe, on="ticker", how="left", validate="many_to_one").copy()
    p = p.dropna(subset=["country"]).copy()
    p["adj_close"] = pd.to_numeric(p["adj_close"], errors="coerce")
    p = p.dropna(subset=["adj_close"]).sort_values(["ticker", "date"]).copy()
    p["ret_d"] = p.groupby("ticker")["adj_close"].pct_change()

    ticker_cov = (
        p.groupby(["country", "ticker", "name"], dropna=False)
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            n_price_obs=("adj_close", "count"),
            n_return_obs=("ret_d", lambda s: int(pd.Series(s).notna().sum())),
        )
        .reset_index()
    )

    d = p.dropna(subset=["ret_d"]).copy()
    country_daily = (
        d.groupby(["country", "date"], dropna=False)
        .agg(
            country_ret_d=("ret_d", "mean"),
            n_tickers=("ticker", "nunique"),
        )
        .reset_index()
        .sort_values(["country", "date"])
    )
    return country_daily, ticker_cov


def aggregate_country_period_returns(country_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = country_daily.copy()
    d["year"] = d["date"].dt.year.astype(int)
    q = d["date"].dt.to_period("Q")
    d["period_str"] = q.astype(str)
    d["quarter_end"] = q.dt.end_time.dt.normalize()

    def agg_fn(g: pd.DataFrame) -> pd.Series:
        r = pd.to_numeric(g["country_ret_d"], errors="coerce").dropna()
        return pd.Series(
            {
                "country_period_ret": float(np.prod(1.0 + r.values) - 1.0) if len(r) else np.nan,
                "n_days": int(len(r)),
                "avg_n_tickers": float(pd.to_numeric(g["n_tickers"], errors="coerce").mean()),
                "min_n_tickers": int(pd.to_numeric(g["n_tickers"], errors="coerce").min()),
                "max_n_tickers": int(pd.to_numeric(g["n_tickers"], errors="coerce").max()),
            }
        )

    annual = d.groupby(["country", "year"], dropna=False).apply(agg_fn).reset_index()
    quarterly = d.groupby(["country", "period_str", "quarter_end"], dropna=False).apply(agg_fn).reset_index()
    annual["country_period_ret_pct"] = annual["country_period_ret"] * 100.0
    quarterly["country_period_ret_pct"] = quarterly["country_period_ret"] * 100.0
    return annual, quarterly


def load_backtest_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_csv(RESULTS_DIR / "investor_style_forecast_predictions.csv")
    annual = p[p["sample"] == "annual_investor_headline"].copy()
    annual["geo"] = annual["geo"].astype(str).str.upper()
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce").astype("Int64")
    annual = annual.dropna(subset=["geo", "year", "yhat"]).copy()

    quarterly = p[p["sample"] == "quarterly_investor_tq_full"].copy()
    quarterly["geo"] = quarterly["geo"].astype(str).str.upper()
    quarterly["period_str"] = quarterly["period_str"].astype(str)
    quarterly = quarterly.dropna(subset=["geo", "period_str", "yhat"]).copy()
    return annual, quarterly


def rank_spread_metrics(
    df: pd.DataFrame,
    period_col: str,
    signal_col: str,
    ret_col: str,
    *,
    tail_frac: float = TAIL_FRAC,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for period, g in df.groupby(period_col):
        g = g.dropna(subset=[signal_col, ret_col]).copy()
        if len(g) < 4:
            continue
        g = g.sort_values(signal_col)
        n = len(g)
        k = max(1, int(np.ceil(n * tail_frac)))
        k = min(k, max(1, n // 2))
        bottom = float(g.iloc[:k][ret_col].mean())
        top = float(g.iloc[-k:][ret_col].mean())
        middle = float(g.iloc[k:n-k][ret_col].mean()) if n - 2 * k > 0 else np.nan
        ic = spearmanr(g[signal_col], g[ret_col], nan_policy="omit").correlation
        rows.append(
            {
                "period": period,
                "n": int(n),
                "k_tail": int(k),
                "tail_frac_target": float(tail_frac),
                "tail_frac_realized": float(k / n) if n else np.nan,
                "rank_ic": float(ic) if ic is not None else np.nan,
                "top_minus_bottom": top - bottom,
                "top_mean": top,
                "bottom_mean": bottom,
                "middle_mean": middle,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["period", "n", "k_tail", "tail_frac_target", "tail_frac_realized", "rank_ic", "top_minus_bottom", "top_mean", "bottom_mean", "middle_mean"]), {
            "periods_eval": 0,
            "mean_rank_ic": np.nan,
            "median_rank_ic": np.nan,
            "mean_top_minus_bottom": np.nan,
            "median_top_minus_bottom": np.nan,
            "mean_k_tail": np.nan,
            "mean_tail_frac_realized": np.nan,
        }
    d = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
    metrics = {
        "periods_eval": int(len(d)),
        "mean_rank_ic": float(pd.to_numeric(d["rank_ic"], errors="coerce").mean(skipna=True)),
        "median_rank_ic": float(pd.to_numeric(d["rank_ic"], errors="coerce").median(skipna=True)),
        "mean_top_minus_bottom": float(pd.to_numeric(d["top_minus_bottom"], errors="coerce").mean(skipna=True)),
        "median_top_minus_bottom": float(pd.to_numeric(d["top_minus_bottom"], errors="coerce").median(skipna=True)),
        "mean_k_tail": float(pd.to_numeric(d["k_tail"], errors="coerce").mean(skipna=True)),
        "mean_tail_frac_realized": float(pd.to_numeric(d["tail_frac_realized"], errors="coerce").mean(skipna=True)),
    }
    return d, metrics


def evaluate_public_market_link(
    preds: pd.DataFrame,
    rets: pd.DataFrame,
    *,
    merge_keys: list[str],
    sample_name: str,
    period_col: str,
) -> EvalOutput:
    d = preds.merge(rets, left_on=["geo"] + merge_keys, right_on=["country"] + merge_keys, how="inner").copy()
    if d.empty:
        return EvalOutput(
            merged=d,
            metrics={"sample": sample_name, "nobs_test": 0},
            period_metrics=pd.DataFrame(),
            ls_period=pd.DataFrame(),
        )

    d["re_public_ret_pct"] = pd.to_numeric(d["country_period_ret_pct"], errors="coerce")
    d["signal_hpi_pred"] = pd.to_numeric(d["yhat"], errors="coerce")
    d["signal_hpi_fe_only"] = pd.to_numeric(d.get("yhat_fe_only"), errors="coerce")
    d["hpi_actual"] = pd.to_numeric(d.get("y"), errors="coerce")

    z = d.dropna(subset=["re_public_ret_pct", "signal_hpi_pred"]).copy()
    err = z["re_public_ret_pct"] - z["signal_hpi_pred"]
    rmse_naive_units = float(np.sqrt(np.mean(err**2))) if len(z) else np.nan
    mae_naive_units = float(np.mean(np.abs(err))) if len(z) else np.nan
    corr_signal_return = _safe_mean_corr(z["signal_hpi_pred"], z["re_public_ret_pct"])
    corr_hpi_actual_return = _safe_mean_corr(z["hpi_actual"], z["re_public_ret_pct"])
    corr_feonly_return = _safe_mean_corr(z["signal_hpi_fe_only"], z["re_public_ret_pct"])

    period_metrics, rank_meta = rank_spread_metrics(
        z,
        period_col=period_col,
        signal_col="signal_hpi_pred",
        ret_col="re_public_ret_pct",
        tail_frac=TAIL_FRAC,
    )
    if "signal_hpi_fe_only" in z.columns:
        fe_period, fe_rank_meta = rank_spread_metrics(
            z,
            period_col=period_col,
            signal_col="signal_hpi_fe_only",
            ret_col="re_public_ret_pct",
            tail_frac=TAIL_FRAC,
        )
    else:
        fe_period = pd.DataFrame()
        fe_rank_meta = {}
    periods_per_year = 1 if period_col == "year" else 4
    perf_signal = strategy_perf_from_period_spreads(period_metrics, periods_per_year=periods_per_year)
    perf_fe = strategy_perf_from_period_spreads(fe_period, periods_per_year=periods_per_year)
    perf_top = strategy_perf_from_period_spreads(period_metrics, spread_col="top_mean", periods_per_year=periods_per_year)
    perf_short = strategy_perf_from_period_spreads(
        period_metrics.assign(short_bottom_mean=-pd.to_numeric(period_metrics.get("bottom_mean"), errors="coerce")),
        spread_col="short_bottom_mean",
        periods_per_year=periods_per_year,
    )

    # Cumulative top-bottom series (periodic arithmetic approximation in percentage points).
    ls = period_metrics.copy()
    if not ls.empty:
        ls = ls.rename(columns={"period": period_col}).copy()
        ls["cum_top_minus_bottom_pp"] = pd.to_numeric(ls["top_minus_bottom"], errors="coerce").cumsum()
        ls["sample"] = sample_name
    if not fe_period.empty:
        fe_period = fe_period.rename(columns={"period": period_col}).copy()
        fe_period["cum_top_minus_bottom_pp"] = pd.to_numeric(fe_period["top_minus_bottom"], errors="coerce").cumsum()
        fe_period["sample"] = sample_name
        fe_period["signal"] = "fe_only"
    if not ls.empty:
        ls["signal"] = "hpi_prediction"

    pm = period_metrics.copy()
    if not pm.empty:
        pm = pm.rename(columns={"period": period_col})
        pm["sample"] = sample_name

    if period_col == "period_str":
        qcol = None
        for candidate in ["quarter_end", "quarter_end_y", "quarter_end_x"]:
            if candidate in z.columns:
                qcol = candidate
                break
        if qcol is not None:
            qmap = z[["period_str", qcol]].dropna().drop_duplicates("period_str").rename(columns={qcol: "quarter_end"})
            if not pm.empty:
                pm = pm.merge(qmap, on="period_str", how="left")
            if not ls.empty:
                ls = ls.merge(qmap, on="period_str", how="left")

    metrics = {
        "sample": sample_name,
        "nobs_test": int(len(z)),
        "countries": int(z["geo"].nunique()),
        "periods_eval": int(rank_meta.get("periods_eval", 0)),
        "corr_pred_hpi_to_re_return": corr_signal_return,
        "corr_actual_hpi_to_re_return": corr_hpi_actual_return,
        "corr_feonly_to_re_return": corr_feonly_return,
        "rmse_signal_vs_re_return_units": rmse_naive_units,
        "mae_signal_vs_re_return_units": mae_naive_units,
        "mean_rank_ic": rank_meta.get("mean_rank_ic", np.nan),
        "median_rank_ic": rank_meta.get("median_rank_ic", np.nan),
        "mean_top_minus_bottom_pp": rank_meta.get("mean_top_minus_bottom", np.nan),
        "median_top_minus_bottom_pp": rank_meta.get("median_top_minus_bottom", np.nan),
        "mean_k_tail": rank_meta.get("mean_k_tail", np.nan),
        "mean_tail_frac_realized": rank_meta.get("mean_tail_frac_realized", np.nan),
        "mean_top_return_pp": float(pd.to_numeric(period_metrics.get("top_mean"), errors="coerce").mean(skipna=True)) if not period_metrics.empty else np.nan,
        "mean_bottom_return_pp": float(pd.to_numeric(period_metrics.get("bottom_mean"), errors="coerce").mean(skipna=True)) if not period_metrics.empty else np.nan,
        "mean_rank_ic_fe_only": fe_rank_meta.get("mean_rank_ic", np.nan),
        "mean_top_minus_bottom_pp_fe_only": fe_rank_meta.get("mean_top_minus_bottom", np.nan),
        "median_top_minus_bottom_pp_fe_only": fe_rank_meta.get("median_top_minus_bottom", np.nan),
        "mean_k_tail_fe_only": fe_rank_meta.get("mean_k_tail", np.nan),
        "annualized_top_bottom_return_pct": perf_signal["annualized_return_pct"],
        "annualized_top_bottom_vol_pct": perf_signal["annualized_vol_pct"],
        "top_bottom_sharpe_0rf": perf_signal["sharpe_0rf"],
        "top_bottom_sortino_0rf": perf_signal["sortino_0rf"],
        "annualized_top_leg_return_pct": perf_top["annualized_return_pct"],
        "annualized_top_leg_vol_pct": perf_top["annualized_vol_pct"],
        "top_leg_sharpe_0rf": perf_top["sharpe_0rf"],
        "top_leg_sortino_0rf": perf_top["sortino_0rf"],
        "annualized_short_leg_return_pct": perf_short["annualized_return_pct"],
        "annualized_short_leg_vol_pct": perf_short["annualized_vol_pct"],
        "short_leg_sharpe_0rf": perf_short["sharpe_0rf"],
        "short_leg_sortino_0rf": perf_short["sortino_0rf"],
        "annualized_top_bottom_return_pct_fe_only": perf_fe["annualized_return_pct"],
        "annualized_top_bottom_vol_pct_fe_only": perf_fe["annualized_vol_pct"],
        "top_bottom_sharpe_0rf_fe_only": perf_fe["sharpe_0rf"],
        "top_bottom_sortino_0rf_fe_only": perf_fe["sortino_0rf"],
    }

    return EvalOutput(merged=d, metrics=metrics, period_metrics=pm, ls_period=ls)


def build_daily_strategy_series(
    preds: pd.DataFrame,
    country_daily: pd.DataFrame,
    *,
    period_col: str,
    sample_name: str,
) -> tuple[pd.DataFrame, dict]:
    if preds.empty or country_daily.empty:
        return pd.DataFrame(), {}

    cd = country_daily.copy()
    cd["country"] = cd["country"].astype(str).str.upper()
    cd["date"] = pd.to_datetime(cd["date"])
    if period_col == "year":
        cd["year"] = cd["date"].dt.year.astype("Int64")
    elif period_col == "period_str":
        cd["period_str"] = cd["date"].dt.to_period("Q").astype(str)
    else:
        raise ValueError(f"Unsupported period_col: {period_col}")

    base_cols = ["geo", period_col, "yhat", "yhat_fe_only"]
    p = preds[base_cols].copy()
    p["geo"] = p["geo"].astype(str).str.upper()

    assign_main = compute_tail_assignments(p.rename(columns={"yhat": "signal"}), period_col=period_col, signal_col="signal", tail_frac=TAIL_FRAC)
    assign_main = assign_main.rename(columns={"signal": "signal_value"})
    assign_main["signal_name"] = "hpi_prediction"

    assign_fe = compute_tail_assignments(p.rename(columns={"yhat_fe_only": "signal"}), period_col=period_col, signal_col="signal", tail_frac=TAIL_FRAC)
    assign_fe = assign_fe.rename(columns={"signal": "signal_value"})
    assign_fe["signal_name"] = "fe_only"

    assigns = pd.concat([assign_main, assign_fe], ignore_index=True, sort=False)
    if assigns.empty:
        return pd.DataFrame(), {}

    d = cd.merge(assigns, left_on=["country", period_col], right_on=["geo", period_col], how="inner")
    d = d.dropna(subset=["country_ret_d"]).copy()
    if d.empty:
        return pd.DataFrame(), {}

    # Daily leg returns per date and signal.
    rows = []
    grp_cols = ["signal_name", "date"]
    if period_col == "year":
        grp_cols.insert(1, "year")
    else:
        grp_cols.insert(1, "period_str")
    for keys, g in d.groupby(grp_cols, dropna=False):
        g = g.copy()
        top = pd.to_numeric(g.loc[g["is_top"], "country_ret_d"], errors="coerce").dropna()
        bot = pd.to_numeric(g.loc[g["is_bottom"], "country_ret_d"], errors="coerce").dropna()
        if top.empty and bot.empty:
            continue
        if period_col == "year":
            signal_name, yr, dt = keys
            row = {"signal_name": signal_name, "year": int(yr), "date": pd.to_datetime(dt)}
        else:
            signal_name, pstr, dt = keys
            row = {"signal_name": signal_name, "period_str": str(pstr), "date": pd.to_datetime(dt)}
        row.update(
            {
                "top_leg_ret_d": float(top.mean()) if len(top) else np.nan,
                "bottom_leg_ret_d": float(bot.mean()) if len(bot) else np.nan,
                "short_bottom_leg_ret_d": float(-bot.mean()) if len(bot) else np.nan,
                "long_short_ret_d": float(top.mean() - bot.mean()) if len(top) and len(bot) else np.nan,
                "n_top_countries_today": int(len(top)),
                "n_bottom_countries_today": int(len(bot)),
            }
        )
        rows.append(row)
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily, {}
    daily["sample"] = sample_name
    daily = daily.sort_values(["signal_name", "date"]).reset_index(drop=True)

    # Cumulative returns (daily).
    for signal_name, g_idx in daily.groupby("signal_name").groups.items():
        idx = list(g_idx)
        for col_in, col_out in [
            ("top_leg_ret_d", "cum_top_leg"),
            ("short_bottom_leg_ret_d", "cum_short_bottom_leg"),
            ("long_short_ret_d", "cum_long_short"),
        ]:
            r = pd.to_numeric(daily.loc[idx, col_in], errors="coerce").fillna(0.0)
            daily.loc[idx, col_out] = (1.0 + r).cumprod() - 1.0

    # Summary metrics (daily-annualized)
    metrics = {"sample": sample_name}
    for sig in ["hpi_prediction", "fe_only"]:
        sub = daily[daily["signal_name"] == sig].copy()
        ls = strategy_perf_from_daily_returns(sub, ret_col="long_short_ret_d")
        topm = strategy_perf_from_daily_returns(sub, ret_col="top_leg_ret_d")
        shortm = strategy_perf_from_daily_returns(sub, ret_col="short_bottom_leg_ret_d")
        prefix = "" if sig == "hpi_prediction" else "fe_only_"
        metrics[f"{prefix}daily_ann_long_short_return_pct"] = ls["ann_return_pct"]
        metrics[f"{prefix}daily_ann_long_short_vol_pct"] = ls["ann_vol_pct"]
        metrics[f"{prefix}daily_long_short_sharpe_0rf"] = ls["sharpe_0rf"]
        metrics[f"{prefix}daily_long_short_sortino_0rf"] = ls["sortino_0rf"]
        metrics[f"{prefix}daily_ann_top_leg_return_pct"] = topm["ann_return_pct"]
        metrics[f"{prefix}daily_ann_top_leg_vol_pct"] = topm["ann_vol_pct"]
        metrics[f"{prefix}daily_top_leg_sharpe_0rf"] = topm["sharpe_0rf"]
        metrics[f"{prefix}daily_ann_short_leg_return_pct"] = shortm["ann_return_pct"]
        metrics[f"{prefix}daily_ann_short_leg_vol_pct"] = shortm["ann_vol_pct"]
        metrics[f"{prefix}daily_short_leg_sharpe_0rf"] = shortm["sharpe_0rf"]
        metrics[f"{prefix}daily_n_days"] = ls["n_days"]
        metrics[f"{prefix}daily_sample_years"] = ls["sample_years"]
    return daily, metrics


def build_coverage_table(
    ticker_cov: pd.DataFrame,
    annual_ret: pd.DataFrame,
    quarterly_ret: pd.DataFrame,
    preds_annual: pd.DataFrame,
    preds_quarterly: pd.DataFrame,
) -> pd.DataFrame:
    cov = (
        ticker_cov.groupby("country", dropna=False)
        .agg(
            tickers_with_prices=("ticker", "nunique"),
            first_date=("first_date", "min"),
            last_date=("last_date", "max"),
            avg_price_obs_per_ticker=("n_price_obs", "mean"),
        )
        .reset_index()
    )
    ann_obs = annual_ret.groupby("country")["year"].nunique().rename("annual_return_years").reset_index()
    q_obs = quarterly_ret.groupby("country")["period_str"].nunique().rename("quarterly_return_periods").reset_index()
    pa = preds_annual.groupby("geo")["year"].nunique().rename("annual_prediction_years").reset_index().rename(columns={"geo": "country"})
    pq = preds_quarterly.groupby("geo")["period_str"].nunique().rename("quarterly_prediction_periods").reset_index().rename(columns={"geo": "country"})
    cov = cov.merge(ann_obs, on="country", how="left").merge(q_obs, on="country", how="left").merge(pa, on="country", how="left").merge(pq, on="country", how="left")
    for c in ["annual_return_years", "quarterly_return_periods", "annual_prediction_years", "quarterly_prediction_periods"]:
        cov[c] = pd.to_numeric(cov[c], errors="coerce").fillna(0).astype(int)
    cov = cov.sort_values(["tickers_with_prices", "country"], ascending=[False, True]).reset_index(drop=True)
    return cov


def _fmt(x, nd=3):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    return f"{float(x):.{nd}f}"


def write_latex_tables(metrics_df: pd.DataFrame, coverage_df: pd.DataFrame) -> None:
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Summary metrics table.
    rows = []
    label_map = {
        "annual_public_re_validation": "Annual public-market translation",
        "quarterly_public_re_validation": "Quarterly public-market translation",
    }
    for _, r in metrics_df.iterrows():
        rows.append(
            [
                label_map.get(r["sample"], r["sample"]),
                _fmt(r["countries"], 0),
                _fmt(r["nobs_test"], 0),
                _fmt(r["periods_eval"], 0),
                _fmt(r["corr_pred_hpi_to_re_return"], 3),
                _fmt(r["mean_rank_ic"], 3),
                _fmt(r["mean_top_minus_bottom_pp"], 2),
                _fmt(r["median_top_minus_bottom_pp"], 2),
                _fmt(r["mean_k_tail"], 1),
                _fmt(r["mean_top_minus_bottom_pp_fe_only"], 2),
            ]
        )
    metrics_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Public-market validation: country-level listed real-estate equity returns sorted by house-price model predictions}",
        r"\label{tab:public_market_validation}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{p{4.2cm}rrrrrrrrr}",
        r"\toprule",
        f"Sample & Countries & Obs. & Periods & Corr(pred., RE ret.) & Mean rank IC & Mean {TAIL_PAIR_LABEL} (pp) & Median {TAIL_PAIR_LABEL} (pp) & Avg names per tail & FE-only mean {TAIL_PAIR_LABEL} (pp) \\\\",
        r"\midrule",
    ]
    for row in rows:
        metrics_lines.append(" {} \\\\".format(" & ".join(row)))
    metrics_lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        f"\\item Notes: The signal is the pseudo-out-of-sample prediction of next-period house-price growth from the paper's expanding-window backtest. We map that signal to country baskets of listed real-estate equities (REITs and listed property companies), built from daily Yahoo Finance \\emph{{adjusted-close}} returns (a dividend-adjusted total-return proxy when provider adjustments are available) and aggregated to calendar year/quarter returns. ``{TAIL_PAIR_LABEL}'' is the long-short return spread from buying the top {TAIL_PCT}\\% of countries (highest predicted signal) and shorting the bottom {TAIL_PCT}\\% (lowest predicted signal) within each period, with tail counts rounded up to whole countries (minimum 1 country per tail).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    (PAPER_TABLES_DIR / "tab_public_market_validation.tex").write_text("\n".join(metrics_lines))

    # Strategy performance table (percentage and risk-adjusted metrics).
    perf_rows = []
    for _, r in metrics_df.iterrows():
        sample_lbl = label_map.get(r["sample"], r["sample"])
        perf_rows.extend(
            [
                [
                    sample_lbl,
                    f"Long top {TAIL_PCT}\\%",
                    _fmt(r.get("daily_ann_top_leg_return_pct"), 2),
                    _fmt(r.get("daily_ann_top_leg_vol_pct"), 2),
                    _fmt(r.get("daily_top_leg_sharpe_0rf"), 2),
                    "",
                ],
                [
                    sample_lbl,
                    f"Short bottom {TAIL_PCT}\\%",
                    _fmt(r.get("daily_ann_short_leg_return_pct"), 2),
                    _fmt(r.get("daily_ann_short_leg_vol_pct"), 2),
                    _fmt(r.get("daily_short_leg_sharpe_0rf"), 2),
                    "",
                ],
                [
                    sample_lbl,
                    f"Long-short {TAIL_PAIR_LABEL}",
                    _fmt(r.get("daily_ann_long_short_return_pct"), 2),
                    _fmt(r.get("daily_ann_long_short_vol_pct"), 2),
                    _fmt(r.get("daily_long_short_sharpe_0rf"), 2),
                    _fmt(r.get("daily_long_short_sortino_0rf"), 2),
                ],
                [
                    sample_lbl,
                    f"FE-only long-short {TAIL_PAIR_LABEL}",
                    _fmt(r.get("fe_only_daily_ann_long_short_return_pct"), 2),
                    _fmt(r.get("fe_only_daily_ann_long_short_vol_pct"), 2),
                    _fmt(r.get("fe_only_daily_long_short_sharpe_0rf"), 2),
                    _fmt(r.get("fe_only_daily_long_short_sortino_0rf"), 2),
                ],
            ]
        )
    perf_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        f"\\caption{{Public-market translation strategy performance (top {TAIL_PCT}\\% long, bottom {TAIL_PCT}\\% short country sorting on the housing signal)}}",
        r"\label{tab:public_market_strategy_perf}",
        r"\begin{threeparttable}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{p{4.0cm}p{2.8cm}rrrrrr}",
        r"\toprule",
        r"Sample & Strategy leg & Ann. return (pp) & Ann. vol (pp) & Sharpe & Sortino & Trading days & Sample years \\",
        r"\midrule",
    ]
    # append days/years by sample to each row for transparency
    row_idx = 0
    for _, r in metrics_df.iterrows():
        for _ in range(4):
            row = perf_rows[row_idx]
            if "FE-only" in row[1]:
                row_ext = row + [_fmt(r.get("fe_only_daily_n_days"), 0), _fmt(r.get("fe_only_daily_sample_years"), 1)]
            else:
                row_ext = row + [_fmt(r.get("daily_n_days"), 0), _fmt(r.get("daily_sample_years"), 1)]
            perf_lines.append(" {} \\\\".format(" & ".join(row_ext)))
            row_idx += 1
    perf_lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        f"\\item Notes: Metrics are computed from \\emph{{daily}} country-basket returns while holding period-level {TAIL_PAIR_LABEL} positions fixed within each forecast period (year for the annual signal, quarter for the quarterly signal). ``Short bottom {TAIL_PCT}\\%'' reports the return on a short position in the bottom tail (i.e., negative of the bottom-tail basket return). Returns are shown in percentage points and annualized using 252 trading days. Sharpe and Sortino assume zero risk-free rate.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    (PAPER_TABLES_DIR / "tab_public_market_strategy_perf.tex").write_text("\n".join(perf_lines))

    # Coverage table (paper-friendly subset sorted by merged relevance).
    cov = coverage_df.copy()
    cov = cov[(cov["tickers_with_prices"] > 0) & ((cov["annual_prediction_years"] > 0) | (cov["quarterly_prediction_periods"] > 0))].copy()
    cov = cov[cov["country"] != "PL"].copy()
    cov = cov.sort_values(["tickers_with_prices", "quarterly_prediction_periods", "annual_prediction_years", "country"], ascending=[False, False, False, True]).head(16)
    cov_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Public-market validation coverage: country-tagged listed real-estate equity baskets}",
        r"\label{tab:public_market_coverage}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrll}",
        r"\toprule",
        r"Country & Tickers & Annual pred. years & Quarterly pred. periods & Quarterly RE periods & First date & Last date \\",
        r"\midrule",
    ]
    for _, r in cov.iterrows():
        cov_lines.append(
            " {} \\\\".format(
                " & ".join(
                    [
                        str(r["country"]),
                        _fmt(r["tickers_with_prices"], 0),
                        _fmt(r["annual_prediction_years"], 0),
                        _fmt(r["quarterly_prediction_periods"], 0),
                        _fmt(r["quarterly_return_periods"], 0),
                        pd.to_datetime(r["first_date"]).strftime("%Y-%m-%d") if pd.notna(r["first_date"]) else "",
                        pd.to_datetime(r["last_date"]).strftime("%Y-%m-%d") if pd.notna(r["last_date"]) else "",
                    ]
                )
            )
        )
    cov_lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Notes: Country baskets are equal-weight averages of available daily adjusted-close returns across the listed real-estate securities in the curated universe. Using adjusted close means the series is a dividend-adjusted total-return proxy when the data provider supplies dividend/split adjustments. Coverage is uneven across countries because listed real-estate sectors differ in size and listing history.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    (PAPER_TABLES_DIR / "tab_public_market_coverage.tex").write_text("\n".join(cov_lines))


def plot_public_market_figures(
    merged_all: pd.DataFrame,
    metrics_df: pd.DataFrame,
    ls_all: pd.DataFrame,
    daily_strategy_all: pd.DataFrame,
) -> None:
    PAPER_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    colors = {
        "annual_public_re_validation": "#1f4e79",
        "quarterly_public_re_validation": "#b22222",
    }
    titles = {
        "annual_public_re_validation": "Annual signal vs next-year listed real-estate return",
        "quarterly_public_re_validation": "Quarterly signal vs next-quarter listed real-estate return",
    }

    # Scatter figure.
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    for ax, sample in zip(axes, ["annual_public_re_validation", "quarterly_public_re_validation"]):
        d = merged_all[merged_all["sample_eval"] == sample].dropna(subset=["signal_hpi_pred", "re_public_ret_pct"]).copy()
        if d.empty:
            ax.text(0.5, 0.5, "No matched observations", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        x = d["signal_hpi_pred"].astype(float)
        y = d["re_public_ret_pct"].astype(float)
        ax.scatter(x, y, s=18, alpha=0.45, color=colors[sample], edgecolor="none")
        # Best-fit line (descriptive).
        if len(d) >= 2:
            b1, b0 = np.polyfit(x, y, 1)
            xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
            ax.plot(xx, b1 * xx + b0, color="#333333", lw=1.2)
        ax.axhline(0, color="#888888", lw=0.8, ls="--")
        ax.axvline(0, color="#888888", lw=0.8, ls=":")
        ax.set_title(titles[sample])
        ax.set_xlabel("Predicted next-period house-price growth (pp)")
        ax.set_ylabel("Listed real-estate basket return in same future period (pp)")
        ax.grid(alpha=0.2)
        m = metrics_df[metrics_df["sample"] == sample]
        if not m.empty:
            r = m.iloc[0]
            txt = (
                f"N={int(r['nobs_test'])}, periods={int(r['periods_eval'])}\n"
                f"Corr(pred., RE ret.)={r['corr_pred_hpi_to_re_return']:.2f}\n"
                f"Mean rank IC={r['mean_rank_ic']:.2f}\n"
                f"Top-bottom={r['mean_top_minus_bottom_pp']:.2f} pp"
            )
            ax.text(
                0.03, 0.97, txt,
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="#bbbbbb", alpha=0.9, boxstyle="round,pad=0.35"),
            )
    fig.suptitle("Public-market translation of the housing signal: predicted house-price growth vs listed real-estate returns", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(PAPER_FIGS_DIR / "fig_public_market_pred_vs_return.pdf", bbox_inches="tight")
    plt.close(fig)

    # Daily cumulative tail strategy figure (signal formed at period rebalance dates).
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    for ax, sample in zip(axes, ["annual_public_re_validation", "quarterly_public_re_validation"]):
        d = daily_strategy_all[daily_strategy_all["sample"] == sample].copy()
        if d.empty:
            ax.text(0.5, 0.5, f"No daily {TAIL_PAIR_LABEL} strategy series", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        main = d[d["signal_name"] == "hpi_prediction"].copy().sort_values("date")
        fe = d[d["signal_name"] == "fe_only"].copy().sort_values("date")
        ax.set_xlabel("Date")
        ax.plot(main["date"], 100.0 * pd.to_numeric(main["cum_long_short"], errors="coerce"), color=colors[sample], lw=2.0, label=f"Housing signal {TAIL_PAIR_LABEL} (L/S)")
        ax.plot(main["date"], 100.0 * pd.to_numeric(main["cum_top_leg"], errors="coerce"), color=colors[sample], lw=1.0, alpha=0.35, ls=":", label=f"Top {TAIL_PCT}% long leg")
        ax.plot(main["date"], 100.0 * pd.to_numeric(main["cum_short_bottom_leg"], errors="coerce"), color=colors[sample], lw=1.0, alpha=0.35, ls="-.", label=f"Bottom {TAIL_PCT}% short leg")
        if not fe.empty:
            ax.plot(fe["date"], 100.0 * pd.to_numeric(fe["cum_long_short"], errors="coerce"), color="#666666", lw=1.5, ls="--", label=f"FE-only {TAIL_PAIR_LABEL} (L/S)")
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_title(f"Daily cumulative {TAIL_PAIR_LABEL} listed RE return" + (" (annual signal)" if sample.startswith("annual") else " (quarterly signal)"))
        ax.set_ylabel("Cumulative return (%, rebased to 0)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(PAPER_FIGS_DIR / "fig_public_market_top_bottom_cum.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_universe()
    prices = fetch_daily_prices(universe)
    if prices.empty:
        raise RuntimeError("No daily prices downloaded from yfinance.")
    prices.to_parquet(PRICES_CACHE, index=False)

    country_daily, ticker_cov = build_country_daily_returns(prices, universe)
    annual_ret, quarterly_ret = aggregate_country_period_returns(country_daily)
    country_daily.to_csv(RESULTS_DIR / "public_market_country_daily_returns.csv", index=False)
    annual_ret.to_csv(RESULTS_DIR / "public_market_country_annual_returns.csv", index=False)
    quarterly_ret.to_csv(RESULTS_DIR / "public_market_country_quarterly_returns.csv", index=False)
    ticker_cov.to_csv(RESULTS_DIR / "public_market_ticker_coverage.csv", index=False)

    preds_annual, preds_quarterly = load_backtest_predictions()

    annual_eval = evaluate_public_market_link(
        preds_annual,
        annual_ret.rename(columns={"country": "country"}),
        merge_keys=["year"],
        sample_name="annual_public_re_validation",
        period_col="year",
    )

    quarterly_eval = evaluate_public_market_link(
        preds_quarterly,
        quarterly_ret.rename(columns={"country": "country"}),
        merge_keys=["period_str"],
        sample_name="quarterly_public_re_validation",
        period_col="period_str",
    )

    metrics_df = pd.DataFrame([annual_eval.metrics, quarterly_eval.metrics])

    period_df = pd.concat(
        [
            annual_eval.period_metrics.assign(sample_eval="annual_public_re_validation"),
            quarterly_eval.period_metrics.assign(sample_eval="quarterly_public_re_validation"),
        ],
        ignore_index=True,
        sort=False,
    )
    period_df.to_csv(RESULTS_DIR / "public_market_validation_period_metrics.csv", index=False)

    merged_all = pd.concat(
        [
            annual_eval.merged.assign(sample_eval="annual_public_re_validation"),
            quarterly_eval.merged.assign(sample_eval="quarterly_public_re_validation"),
        ],
        ignore_index=True,
        sort=False,
    )
    merged_all.to_csv(RESULTS_DIR / "public_market_validation_merged.csv", index=False)

    # Build daily-held strategy returns from period-level signals (annual and quarterly rebalances).
    annual_daily, annual_daily_meta = build_daily_strategy_series(
        preds_annual,
        country_daily,
        period_col="year",
        sample_name="annual_public_re_validation",
    )
    quarterly_daily, quarterly_daily_meta = build_daily_strategy_series(
        preds_quarterly,
        country_daily,
        period_col="period_str",
        sample_name="quarterly_public_re_validation",
    )
    daily_strategy_all = pd.concat([annual_daily, quarterly_daily], ignore_index=True, sort=False)
    daily_strategy_all.to_csv(RESULTS_DIR / "public_market_validation_daily_strategy_returns.csv", index=False)

    daily_meta_df = pd.DataFrame([annual_daily_meta, quarterly_daily_meta])
    if not daily_meta_df.empty:
        metrics_df = metrics_df.merge(daily_meta_df, on="sample", how="left")
    metrics_df.to_csv(RESULTS_DIR / "public_market_validation_metrics.csv", index=False)
    metrics_df.to_csv(RESULTS_DIR / "public_market_validation_strategy_metrics.csv", index=False)

    ls_all = pd.concat(
        [
            annual_eval.ls_period,
            quarterly_eval.ls_period,
        ],
        ignore_index=True,
        sort=False,
    )
    # Add FE-only cumulative spread series.
    # Recompute from merged data for clarity.
    extra_ls = []
    for sample_name, d in [("annual_public_re_validation", annual_eval.merged), ("quarterly_public_re_validation", quarterly_eval.merged)]:
        if d.empty:
            continue
        if sample_name.startswith("annual"):
            period_col = "year"
        else:
            period_col = "period_str"
        fe_pm, _ = rank_spread_metrics(
            d.dropna(subset=["signal_hpi_fe_only", "re_public_ret_pct"]),
            period_col=period_col,
            signal_col="signal_hpi_fe_only",
            ret_col="re_public_ret_pct",
            tail_frac=TAIL_FRAC,
        )
        if fe_pm.empty:
            continue
        fe_pm = fe_pm.rename(columns={"period": period_col}).copy()
        fe_pm["cum_top_minus_bottom_pp"] = pd.to_numeric(fe_pm["top_minus_bottom"], errors="coerce").cumsum()
        fe_pm["sample"] = sample_name
        fe_pm["signal"] = "fe_only"
        if period_col == "period_str":
            qcol = None
            for candidate in ["quarter_end", "quarter_end_y", "quarter_end_x"]:
                if candidate in d.columns:
                    qcol = candidate
                    break
            if qcol is not None:
                quarter_map = d[["period_str", qcol]].dropna().drop_duplicates("period_str").rename(columns={qcol: "quarter_end"})
                fe_pm = fe_pm.merge(quarter_map, on="period_str", how="left")
        extra_ls.append(fe_pm)
    if extra_ls:
        ls_all = pd.concat([ls_all] + extra_ls, ignore_index=True, sort=False)
    ls_all.to_csv(RESULTS_DIR / "public_market_validation_longshort_period.csv", index=False)

    coverage_df = build_coverage_table(ticker_cov, annual_ret, quarterly_ret, preds_annual, preds_quarterly)
    coverage_df.to_csv(RESULTS_DIR / "public_market_validation_country_coverage.csv", index=False)

    write_latex_tables(metrics_df, coverage_df)
    plot_public_market_figures(merged_all, metrics_df, ls_all, daily_strategy_all)

    meta = {
        "ticker_universe_count": int(universe["ticker"].nunique()),
        "tickers_with_prices": int(prices["ticker"].nunique()),
        "countries_with_country_daily_returns": int(country_daily["country"].nunique()),
        "annual_public_re_validation": annual_eval.metrics,
        "quarterly_public_re_validation": quarterly_eval.metrics,
        "annual_public_re_validation_daily_strategy": annual_daily_meta,
        "quarterly_public_re_validation_daily_strategy": quarterly_daily_meta,
    }
    (META_DIR / "public_market_validation_summary.json").write_text(json.dumps(meta, indent=2))

    print(f"[ok] wrote {RESULTS_DIR / 'public_market_validation_metrics.csv'}")
    print(f"[ok] wrote {RESULTS_DIR / 'public_market_validation_country_coverage.csv'}")
    print(f"[ok] wrote {PAPER_TABLES_DIR / 'tab_public_market_validation.tex'}")
    print(f"[ok] wrote {PAPER_FIGS_DIR / 'fig_public_market_pred_vs_return.pdf'}")


if __name__ == "__main__":
    main()
