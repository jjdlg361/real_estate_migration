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
FIG_DIR = ROOT / "paper_overleaf" / "figures"

OUT_MAP = FIG_DIR / "fig_movement_dynamics_country_corr_map.pdf"
OUT_PHASE = FIG_DIR / "fig_movement_dynamics_phase_portraits.pdf"


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


def plot_corr_map(df: pd.DataFrame) -> None:
    q = df.copy()
    q["hpi_yoy"] = pd.to_numeric(q.get("hpi_yoy_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(q.get("hpi_yoy"), errors="coerce")
    )
    q["air_yoy"] = pd.to_numeric(q.get("air_yoy"), errors="coerce")
    q = q.dropna(subset=["geo", "hpi_yoy", "air_yoy"]).copy()

    rows = []
    for geo, g in q.groupby("geo", sort=False):
        if len(g) < 24:
            continue
        c = g["hpi_yoy"].corr(g["air_yoy"])
        if pd.isna(c):
            continue
        rows.append({"geo": str(geo), "corr_air_hpi_q": float(c), "n_obs": int(len(g))})
    d = pd.DataFrame(rows)
    geoms = load_country_polygons()

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    vals = d.set_index("geo")["corr_air_hpi_q"]
    countries = [c for c in vals.index if c in geoms]
    v = vals.loc[countries]
    vmax = max(0.05, float(np.nanmax(np.abs(v.values))) if len(v) else 0.05)
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    patches = []
    colors = []
    for c in countries:
        for coords in _iter_exterior_coords(geoms[c]):
            patches.append(MplPolygon(coords[:, :2], closed=True))
            colors.append(float(vals.loc[c]))
    if patches:
        pc = PatchCollection(patches, cmap=cmap, norm=norm, linewidths=0.35, edgecolor="#FFFFFF")
        pc.set_array(np.asarray(colors))
        ax.add_collection(pc)
        cbar = plt.colorbar(pc, ax=ax, fraction=0.04, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Correlation: quarterly air YoY vs house-price YoY", fontsize=8)

    missing = [c for c in geoms if c not in countries]
    miss = []
    for c in missing:
        for coords in _iter_exterior_coords(geoms[c]):
            miss.append(MplPolygon(coords[:, :2], closed=True))
    if miss:
        pc_m = PatchCollection(miss, facecolor="#F1F1F1", edgecolor="#DDDDDD", linewidths=0.25)
        ax.add_collection(pc_m)

    ax.set_title("Movement Dynamics Map: Air-Mobility vs Housing Co-Movement (Quarterly)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-25, 40)
    ax.set_ylim(34, 72)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.tight_layout()
    fig.savefig(OUT_MAP, bbox_inches="tight")
    plt.close(fig)


def plot_phase_portraits(df: pd.DataFrame) -> None:
    q = df.copy()
    q["hpi_yoy"] = pd.to_numeric(q.get("hpi_yoy_harmonized"), errors="coerce").combine_first(
        pd.to_numeric(q.get("hpi_yoy"), errors="coerce")
    )
    q["air_yoy"] = pd.to_numeric(q.get("air_yoy"), errors="coerce")
    q["year"] = pd.to_numeric(q.get("year"), errors="coerce")
    q = q.dropna(subset=["geo", "hpi_yoy", "air_yoy", "year"]).copy()

    nobs = q.groupby("geo").size().sort_values(ascending=False)
    focus = nobs.head(6).index.tolist()
    d = q[q["geo"].isin(focus)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6), sharex=True, sharey=True)
    axes = axes.ravel()
    cmap = plt.get_cmap("viridis")
    year_min = int(d["year"].min())
    year_max = int(d["year"].max())
    norm = plt.Normalize(vmin=year_min, vmax=year_max)

    for i, geo in enumerate(focus):
        ax = axes[i]
        g = d[d["geo"] == geo].sort_values(["year", "quarter"]).copy()
        x = g["air_yoy"].to_numpy()
        y = g["hpi_yoy"].to_numpy()
        t = g["year"].to_numpy()
        for j in range(1, len(g)):
            ax.plot([x[j - 1], x[j]], [y[j - 1], y[j]], color=cmap(norm(t[j])), lw=1.2, alpha=0.75)
        sc = ax.scatter(x, y, c=t, cmap=cmap, norm=norm, s=14, alpha=0.9, edgecolor="none")
        ax.axhline(0, color="#777777", lw=0.7)
        ax.axvline(0, color="#777777", lw=0.7)
        ax.set_title(str(geo), fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.2)

    for k in range(len(focus), len(axes)):
        axes[k].axis("off")

    fig.suptitle("Movement Dynamics Phase Portraits: Quarterly Air vs Housing Trajectories", fontsize=13, y=0.98)
    fig.supxlabel("Air-passenger YoY growth (%)")
    fig.supylabel("House-price YoY growth (%)")
    cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Year")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PHASE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    q = pd.read_parquet(PROC_DIR / "panel_quarterly_harmonized.parquet")
    plot_corr_map(q)
    plot_phase_portraits(q)
    print(f"[ok] wrote {OUT_MAP}")
    print(f"[ok] wrote {OUT_PHASE}")


if __name__ == "__main__":
    main()
