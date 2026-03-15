# Do Migration and Air Mobility Raise House Prices in Europe?

Replication package for the paper analysis.

## Package contents

- `scripts/` — full data, estimation, diagnostics, and plotting pipeline
- `data/` — raw and processed datasets used by the pipeline
- `results/` — model outputs, diagnostics, projections, and CSV tables
- `docs/` — research design and supporting documentation
- `requirements.txt` — Python dependencies

## Environment

- Python `3.9+`
- Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Core replication run

From the package root:

```bash
python3 scripts/harmonize_cross_frequency.py
python3 scripts/estimate_models.py
python3 scripts/estimate_models_harmonized.py
python3 scripts/estimate_migration_composition.py
python3 scripts/estimate_dehaas_factors.py
python3 scripts/estimate_traveler_quality_proxies.py
python3 scripts/estimate_expanded_channels.py
python3 scripts/estimate_country_web_extensions.py
python3 scripts/estimate_release_aware_predictive.py
python3 scripts/estimate_model_fit_decomposition.py
python3 scripts/build_model_fit_paper_tables.py
python3 scripts/rebuild_paper_tables_audited.py
python3 scripts/plot_t1_prediction_accuracy.py
python3 scripts/plot_next_year_country_projection_map.py
python3 scripts/plot_next_year_projection_visuals.py
```

## Build the paper PDF

```bash
cd paper_overleaf
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Main outputs:

- Main coefficient tables: `paper_overleaf/tables/`
- Main figures: `paper_overleaf/figures/`

## Key result files

- Baseline and FE models: `results/model_coefficients.csv`, `results/model_summaries.txt`
- Migration composition models: `results/migration_composition_coefficients.csv`
- Migration-system factor models: `results/dehaas_factors_coefficients.csv`
- Release-aware predictive metrics: `results/release_aware_metrics.csv`
- Annual t+1 diagnostics: `results/t1_prediction_diagnostics_annual.csv`
- Next-year country projections: `results/next_year_house_price_projection_country.csv`

## Reproducibility notes

- The pipeline uses harmonized annual and quarterly inputs and source-specific blending where implemented.
- Some country-series endpoints have publication lags; release-aware predictive scripts account for timing constraints.
- Regional and IV sections are included as reported in manuscript tables and diagnostics.
