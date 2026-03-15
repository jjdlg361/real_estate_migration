#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

try:
    from scipy.stats import chi2  # type: ignore
except Exception:  # pragma: no cover
    chi2 = None


ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


def winsorize(s: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.quantile([p_low, p_high])
    return x.clip(lo, hi)


def _first_event_map(
    df: pd.DataFrame,
    event_flag_col: str,
    min_lead: int = 8,
    min_lag: int = 8,
) -> pd.Series:
    event_q_by_geo: dict[str, int] = {}
    for geo, g in df.groupby("geo", sort=False):
        g = g.sort_values("quarter_id").copy()
        qids = g["quarter_id"].to_numpy()
        flags = g[event_flag_col].fillna(0).astype(int).to_numpy()
        if flags.sum() == 0:
            continue
        for i, (qid, flag) in enumerate(zip(qids, flags)):
            if flag != 1:
                continue
            if i < min_lead or (len(g) - i - 1) < min_lag:
                continue
            if flags[max(0, i - min_lead):i].sum() > 0:
                continue
            event_q_by_geo[geo] = int(qid)
            break
    return pd.Series(event_q_by_geo, name="event_qid")


def _make_event_dummies(df: pd.DataFrame, prefix: str, min_h: int = -8, max_h: int = 8, omit: int = -1) -> tuple[pd.DataFrame, list[str], list[str]]:
    d = df.copy()
    d["event_time"] = d["quarter_id"] - d["event_qid"]
    lead_terms: list[str] = []
    all_terms: list[str] = []
    for h in range(min_h, max_h + 1):
        if h == omit:
            continue
        name = f"{prefix}_{'m'+str(abs(h)) if h < 0 else ('p'+str(h) if h > 0 else '0')}"
        d[name] = ((d["event_time"] == h) & d["event_qid"].notna()).astype(int)
        all_terms.append(name)
        if h <= -2:
            lead_terms.append(name)
    return d, all_terms, lead_terms


def _wald_joint_zero(res, terms: list[str]) -> dict[str, float | int | None]:
    if not terms:
        return {"stat": None, "df": 0, "p_value": None}
    cov = res.cov.loc[terms, terms]
    beta = res.params.loc[terms].to_numpy().reshape(-1, 1)
    V = cov.to_numpy()
    stat_arr = beta.T @ np.linalg.pinv(V) @ beta
    stat = float(np.asarray(stat_arr).reshape(1)[0])
    # Numerical noise can produce tiny negative values with pseudo-inverse covariance matrices.
    stat = max(stat, 0.0)
    df = len(terms)
    p_value = None
    if chi2 is not None:
        p_value = float(chi2.sf(stat, df))
    return {"stat": stat, "df": df, "p_value": p_value}


def run_event_study(df: pd.DataFrame, event_type: str) -> tuple[pd.DataFrame, str, dict[str, object]]:
    work = df.copy()
    work["rhpi_yoy"] = winsorize(work["rhpi_yoy"], 0.01, 0.99)
    work = work.sort_values(["geo", "quarter_id"]).reset_index(drop=True)
    if "L1_rhpi_yoy" not in work.columns:
        work["L1_rhpi_yoy"] = work.groupby("geo", sort=False)["rhpi_yoy"].shift(1)
    work["L1_rhpi_yoy"] = winsorize(work["L1_rhpi_yoy"], 0.01, 0.99)
    work["open_persist_rate_norm_q"] = winsorize(work["open_persist_rate_norm_q"], 0.01, 0.99)
    work["close_persist_rate_norm_q"] = winsorize(work["close_persist_rate_norm_q"], 0.01, 0.99)

    if event_type == "open":
        positive = work["open_persist_rate_norm_q"][work["open_persist_rate_norm_q"] > 0]
        thresh = float(positive.quantile(0.75)) if len(positive) else np.nan
        thresh = max(thresh, 0.01) if not math.isnan(thresh) else 0.01
        work["event_flag"] = (
            (work["route_open_persist_q"].fillna(0) >= 1)
            & (work["open_persist_rate_norm_q"].fillna(0) >= thresh)
        ).astype(int)
        prefix = "evt_open"
        label = "regional_event_study_open_persistent"
    elif event_type == "close":
        positive = work["close_persist_rate_norm_q"][work["close_persist_rate_norm_q"] > 0]
        thresh = float(positive.quantile(0.75)) if len(positive) else np.nan
        thresh = max(thresh, 0.01) if not math.isnan(thresh) else 0.01
        work["event_flag"] = (
            (work["route_close_persist_q"].fillna(0) >= 1)
            & (work["close_persist_rate_norm_q"].fillna(0) >= thresh)
        ).astype(int)
        prefix = "evt_close"
        label = "regional_event_study_close_persistent"
    else:
        raise ValueError(event_type)

    first_events = _first_event_map(work[["geo", "quarter_id", "event_flag"]], "event_flag", min_lead=8, min_lag=8)
    work = work.merge(first_events, left_on="geo", right_index=True, how="left")
    work, terms, lead_terms = _make_event_dummies(work, prefix=prefix, min_h=-8, max_h=8, omit=-1)

    sample = work.dropna(subset=["rhpi_yoy", "L1_rhpi_yoy"]).copy()
    panel = sample.set_index(["geo", "quarter_id"]).sort_index()
    formula = "rhpi_yoy ~ 1 + L1_rhpi_yoy + " + " + ".join(terms) + " + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(formula, data=panel, drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    coef_rows = []
    for term in terms:
        if term not in res.params.index:
            continue
        # Parse event time from term name for plotting / paper tables.
        suffix = term.split("_")[-1]
        if suffix == "0":
            h = 0
        elif suffix.startswith("m"):
            h = -int(suffix[1:])
        elif suffix.startswith("p"):
            h = int(suffix[1:])
        else:
            h = np.nan
        coef_rows.append(
            {
                "model": label,
                "term": term,
                "event_time": h,
                "coef": float(res.params[term]),
                "std_err": float(res.std_errors[term]),
                "t_stat": float(res.tstats[term]),
                "p_value": float(res.pvalues[term]),
                "nobs": float(res.nobs),
            }
        )
    coef_df = pd.DataFrame(coef_rows).sort_values("event_time")
    pretrend = _wald_joint_zero(res, [t for t in lead_terms if t in res.params.index])
    info = {
        "model": label,
        "nobs": float(res.nobs),
        "regions_total": int(sample["geo"].nunique()),
        "treated_regions": int(sample.loc[sample["event_qid"].notna(), "geo"].nunique()),
        "event_threshold_rate": thresh,
        "pretrend_joint_test": pretrend,
    }
    text = f"# {label}\nFormula: {formula}\n\n{res.summary}\n\nPretrend joint test (leads=-8..-2): {pretrend}"
    return coef_df, text, info


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROC_DIR / "panel_nuts2_quarterly_route_shocks.parquet").replace([np.inf, -np.inf], np.nan)

    coef_frames = []
    texts = []
    infos = []
    for event_type in ["open", "close"]:
        coef, text, info = run_event_study(df, event_type)
        coef_frames.append(coef)
        texts.append(text)
        infos.append(info)

    coef_all = pd.concat(coef_frames, ignore_index=True)
    coef_all.to_csv(RESULTS_DIR / "event_study_pretrend_coefficients.csv", index=False)
    (RESULTS_DIR / "event_study_pretrend_summary.json").write_text(json.dumps(infos, indent=2))
    (RESULTS_DIR / "event_study_pretrend_summaries.txt").write_text("\n\n" + ("\n\n" + "=" * 100 + "\n\n").join(texts))
    print(json.dumps(infos, indent=2))
    print("[done] Event-study pre-trend models estimated")


if __name__ == "__main__":
    main()
