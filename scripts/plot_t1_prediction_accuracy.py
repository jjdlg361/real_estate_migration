#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PAPER_FIG_DIR = ROOT / "paper_overleaf" / "figures"
PAPER_TAB_DIR = ROOT / "paper_overleaf" / "tables"


def _metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    err = y - p
    rmse = float(np.sqrt(np.nanmean(np.square(err))))
    mae = float(np.nanmean(np.abs(err)))
    corr = float(pd.concat([y.rename("y"), p.rename("p")], axis=1).corr().iloc[0, 1])
    return {"rmse": rmse, "mae": mae, "corr": corr}


def _plot_panel(ax, x: pd.Series, y: pd.Series, title: str, stats: dict[str, float], xlim: tuple[float, float]) -> None:
    ax.scatter(x, y, s=20, alpha=0.65, color="#1f77b4", edgecolor="none")
    ax.plot(xlim, xlim, color="#222222", lw=1.3, linestyle="--", label="45-degree")
    if len(x) >= 2 and x.std() > 0:
        b1, b0 = np.polyfit(x.values, y.values, 1)
        xx = np.linspace(xlim[0], xlim[1], 100)
        ax.plot(xx, b1 * xx + b0, color="#c0392b", lw=1.4, label="Fit line")
    ax.set_title(title)
    ax.set_xlabel("Actual house-price growth (t+1), pp")
    ax.set_ylabel("Predicted house-price growth, pp")
    ax.set_xlim(*xlim)
    ax.set_ylim(*xlim)
    ax.grid(alpha=0.25)
    txt = f"RMSE: {stats['rmse']:.3f}\nMAE: {stats['mae']:.3f}\nCorr: {stats['corr']:.3f}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#dddddd"})


def write_table(metrics_df: pd.DataFrame) -> None:
    PAPER_TAB_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Out-of-sample annual t+1 prediction diagnostics}",
        r"\label{tab:t1_oos_diag}",
        r"\begin{threeparttable}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Obs. & RMSE & MAE & Corr(actual,pred) \\",
        r"\midrule",
    ]
    for _, r in metrics_df.iterrows():
        lines.append(
            f"{r['model_label']} & {int(r['nobs'])} & {r['rmse']:.3f} & {r['mae']:.3f} & {r['corr']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item Notes: Backtest sample is the annual release-aware t+1 setup. The full model includes migration, composition, and macro controls under release-aware timing. Metrics compare model predictions against realized next-year house-price growth.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
            "",
        ]
    )
    (PAPER_TAB_DIR / "tab_t1_prediction_diagnostics.tex").write_text("\n".join(lines))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(RESULTS_DIR / "release_aware_predictions.csv")
    pred = pred[pred["sample"] == "annual_release_aware"].copy()
    for c in ["y", "yhat", "yhat_fe_only"]:
        pred[c] = pd.to_numeric(pred[c], errors="coerce")
    pred = pred.dropna(subset=["y", "yhat", "yhat_fe_only"]).copy()

    y = pred["y"]
    yhat_full = pred["yhat"]
    yhat_fe = pred["yhat_fe_only"]

    m_full = _metrics(y, yhat_full)
    m_fe = _metrics(y, yhat_fe)

    metrics_df = pd.DataFrame(
        [
            {"model": "fe_only", "model_label": "FE-only", "nobs": len(pred), **m_fe},
            {"model": "release_aware", "model_label": "Release-aware full model", "nobs": len(pred), **m_full},
        ]
    )
    metrics_df.to_csv(RESULTS_DIR / "t1_prediction_diagnostics_annual.csv", index=False)

    lo = float(np.nanmin([y.min(), yhat_full.min(), yhat_fe.min()]))
    hi = float(np.nanmax([y.max(), yhat_full.max(), yhat_fe.max()]))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True, sharey=True)
    _plot_panel(axes[0], y, yhat_fe, "FE-only benchmark", m_fe, lim)
    _plot_panel(axes[1], y, yhat_full, "Release-aware full model", m_full, lim)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Out-of-sample Annual t+1 Prediction Accuracy", y=1.02, fontsize=13)
    fig.tight_layout()

    out_pdf = PAPER_FIG_DIR / "fig_t1_prediction_accuracy.pdf"
    out_png = RESULTS_DIR / "fig_t1_prediction_accuracy.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")

    write_table(metrics_df)

    print(f"[ok] wrote {out_pdf}")
    print(f"[ok] wrote {out_png}")
    print(f"[ok] wrote {RESULTS_DIR / 't1_prediction_diagnostics_annual.csv'}")
    print(f"[ok] wrote {PAPER_TAB_DIR / 'tab_t1_prediction_diagnostics.tex'}")


if __name__ == "__main__":
    main()
