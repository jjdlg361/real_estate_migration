# Research Design Memo: Air Mobility, Migration, and House Prices in Europe

## Objective

Build a reproducible Eurostat-based dataset and empirical strategy to study whether cross-border mobility (migration and flight traffic) is associated with residential real estate price growth across European countries.

This repo now includes:

- a data pull pipeline from Eurostat,
- cleaned annual and quarterly panel datasets,
- baseline two-way fixed-effects models,
- an OD migration shift-share IV input pipeline,
- an asylum-based shift-share IV variant (first-time asylum applicants by origin),
- a monthly airport-route shock pipeline (aggregated to quarterly treatments),
- an airport-to-NUTS2 crosswalk (airport coordinates -> regional treatment mapping),
- a regional NUTS2 panel (OECD RHPI + Eurostat regional mobility/macros),
- a regional NUTS2 quarterly route-shock panel for event-study estimation,
- event-study pre-trend tests for persistent route openings/closures,
- a documented roadmap to upgrade the design into a journal-quality causal paper.

## What We Pulled (Eurostat)

Core outcome:

- `PRC_HPI_A` (annual house price index, all dwellings, 2015=100; annual growth)
- `PRC_HPI_Q` (quarterly house price index, all dwellings, 2015=100; q/q growth)

Mobility / migration regressors:

- `AVIA_PAOC` (air passenger transport by country; annual + quarterly passengers carried)
- `TPS00019` (crude rate of net migration plus adjustment, per 1,000 persons)
- `TPS00176` (immigration, persons)
- `TPS00177` (emigration, persons)

Macro controls (annual panel):

- `TEC00115` (real GDP per capita growth)
- `UNE_RT_A` (unemployment rate, total, 15-74)
- `PRC_HICP_AIND` (HICP inflation, all-items)
- `TPS00001` (population on 1 January)
- `IRT_LT_MCBY_A` (long-term interest rates / Maastricht bond yields)

## Current Coverage (Built Dataset)

Target sample currently includes 29 European countries (EU members in the HPI table plus `IS`, `NO`, `UK`; `TR` excluded for comparability and extreme inflation dynamics).

Raw series coverage:

- Air passengers (`AVIA_PAOC`, annual/quarterly): back to 1993 for many countries
- House price index (`PRC_HPI_Q` / `PRC_HPI_A`): typically starts around 2005 in Eurostat
- Net migration rate (`TPS00019` in this pull): 2013-2024, which drives the annual common sample

Model-ready common samples from this run:

- Annual migration + flights baseline: `2014-2024`, `29` countries, `314` observations
- Annual full-controls model: `2016-2024`, `26` countries, `229` observations
- Quarterly flights baseline: `2006Q1-2025Q3`, `29` countries, `2108` observations

Advanced datasets built in this update:

- **Shift-share IV inputs (country-year)** from `MIGR_IMM5PRV`: `567` country-year rows, `26` destinations, `242` origin countries/groups surviving 2-letter-country filters (`1998-2024`)
- **Asylum-IV inputs (country-year)** from `MIGR_ASYAPPCTZA`: `357` country-year rows with non-missing asylum shift-share exposure
- **Monthly airport-route shock panel** from `AVIA_PAOAC` (airport x partner-country x month): `848,376` route-month rows after filtering (`2004-2025`)
- **Country-quarter airport shock treatments**: `2,469` rows, `29` countries (`2004Q1-2025Q4`)
- **Airport-to-NUTS2 crosswalk**: `425` airports in route data, `411` mapped to NUTS2 (`96.7%` match rate)
- **Regional NUTS2 annual panel** (OECD RHPI + Eurostat NUTS2 covariates): `6,873` rows, `78` regions in `10` countries (`1992-2024`)
- **Regional NUTS2 quarterly route-shock panel** (OECD RHPI + airport-shock mapping): `21,026` rows, `44` regions in `9` countries (`1992Q1-2025Q3`)

See:

- `data/metadata/shiftshare_iv_summary.json`
- `data/metadata/flight_shock_build_summary.json`
- `data/metadata/airport_nuts2_crosswalk_summary.json`
- `data/metadata/regional_panel_summary.json`
- `data/metadata/regional_route_quarterly_panel_summary.json`

## Baseline Models (Implemented)

### 1. Annual country panel (two-way FE)

Dependent variable:

- Annual house-price growth (`PRC_HPI_A`, `RCH_A_AVG`)

Main regressors (lagged one year):

- Net migration rate per 1,000 (`TPS00019`)
- Air passenger growth (log-difference of annual `AVIA_PAOC` passengers)

Controls (lagged one year):

- Real GDP per capita growth
- Unemployment rate
- HICP inflation
- Long-term interest rate
- Population growth

Specification:

- Country fixed effects + year fixed effects
- Clustered SE by country

### 2. Quarterly country panel (two-way FE)

Dependent variable:

- House-price YoY growth (constructed from quarterly HPI index logs)

Main regressor:

- Air-passenger YoY growth (constructed from quarterly `AVIA_PAOC` logs), with lags

Specification:

- Country fixed effects + quarter-time fixed effects
- Clustered SE by country

## Why Quarterly for House Prices, and Why Monthly for Flight Shocks?

Short answer:

- **Eurostat house-price indices (`PRC_HPI_Q`) are quarterly**, not monthly.
- So the main price-outcome regressions must be quarterly (or annual) unless we change the price source.

What we implemented to improve timing anyway:

- We **do use monthly data** for the route/airport shock construction (`AVIA_PAOAC`, monthly airport-partner traffic).
- We detect route openings/closures at monthly frequency and then aggregate to quarter for direct alignment with the quarterly house-price outcome.

Why this is better than forcing a monthly main regression right now:

- avoids mixed-frequency outcome problems for the national Eurostat HPI,
- preserves sharp event timing in the treatment construction,
- keeps the core panel design interpretable and review-friendly.

When monthly outcome regressions become feasible:

- use **OECD RHPI monthly regional series** (implemented in the regional panel ingest), or
- use a private transaction-level/monthly regional house-price source.

## Baseline Results (Association, Not Causal)

From `results/model_coefficients.csv`:

- Annual FE model: lagged net migration rate is positive (`~0.106` pp next-year house-price growth for +1 migrant per 1,000), borderline significant (`p≈0.054`).
- Adding annual flight growth leaves the migration coefficient essentially unchanged; flight growth is not statistically significant in the annual model.
- In the annual full-controls model, the migration coefficient remains positive (`~0.103`, `p≈0.089`), while lagged unemployment is significantly negative.
- Quarterly FE models show a positive association between lagged air-passenger YoY growth and house-price YoY growth:
  - `L1_air_yoy ≈ 0.029` (`p≈0.012`) in the single-lag model
  - `L1_air_yoy ≈ 0.023` (`p≈0.006`) and `L2_air_yoy ≈ 0.015` (`p≈0.069`) in the two-lag model

Interpretation:

- These are reduced-form correlations after FE adjustment.
- They are not yet causal because migration and mobility are endogenous to local economic conditions and housing demand.

## Advanced Extensions Implemented (This Update)

### A. Shift-Share IV Dataset (Country-Year)

Implemented in:

- `scripts/fetch_advanced_sources.py`
- `scripts/build_shiftshare_iv.py`

What is constructed:

- destination-origin immigration panel from `MIGR_IMM5PRV`
- **base-period destination origin shares** (default `2005-2009`)
- **leave-one-out origin-supply shock** instrument component using Europe-wide inflows by origin
- **World Bank push-shock instrument** component (GDP-per-capita downturn, conflict intensity, unemployment; annualized and standardized)
- **asylum-specific shift-share instrument** using `MIGR_ASYAPPCTZA` first-time asylum applicants by origin (leave-one-out, log-difference variant)

Output files:

- `data/processed/shiftshare_country_year_iv.csv`
- `data/processed/panel_annual_iv.parquet`

### B. Monthly Airport-Route Shock Dataset (Event-Study Input)

Implemented in:

- `scripts/build_flight_shocks.py`

Construction summary:

- source: monthly `AVIA_PAOAC` (airport x partner-country passengers)
- treatment event logic at **airport-partner route-month** level:
  - route opening = active this month after inactivity
  - route closure (candidate) = inactive this month after prior activity
  - persistent opening = opening survives in subsequent months (future activity check)
  - persistent closure = closure candidate without near-term reactivation
- aggregated to:
  - airport-month shocks
  - country-quarter shock intensity for direct merge with quarterly HPI panels

Output files:

- `data/processed/airport_partner_route_monthly.parquet`
- `data/processed/airport_monthly_route_shocks.parquet`
- `data/processed/country_quarterly_airport_route_shocks.parquet`
- `data/processed/panel_quarterly_airport_shocks.parquet`

### C. Airport-to-NUTS2 Crosswalk (Regional Treatment Mapping)

Implemented in:

- `scripts/build_airport_nuts2_crosswalk.py`

Construction summary:

- airport list is extracted from the route-month panel (`rep_airp`)
- coordinates are matched using **OurAirports** (`gps_code` / `ident`, non-closed airports)
- airport points are spatially assigned to **GISCO NUTS2 polygons** (point-in-polygon; within-country nearest-polygon fallback)

Output files:

- `data/metadata/airport_nuts2_crosswalk.csv`
- `data/metadata/airport_nuts2_crosswalk_summary.json`

### D. Regional NUTS2 Panel (Publication-Oriented Upgrade)

Implemented in:

- `scripts/build_regional_panel.py`

Data combination:

- **OECD RHPI** regional house prices (`DSD_RHPI@DF_RHPI_ALL`)
- Eurostat NUTS2:
  - air passengers (`TGS00077`)
  - net migration rate (`TGS00099`)
  - GDP (`TGS00003`)
  - unemployment (`TGS00010`)
  - population (`TGS00096`)
  - disposable income (`TGS00026`)

Current limitation:

- Coverage overlap is currently strongest for `10` countries in the automated NUTS2 panel.
- This is still a meaningful regional FE testbed, but a broader publication sample will likely require adding another regional price source or harmonized regional control sources.

### E. Regional NUTS2 Quarterly Route-Shock Panel + Event-Study Pre-Trend Tests

Implemented in:

- `scripts/build_regional_route_quarterly_panel.py`
- `scripts/estimate_event_study_pretrends.py`

Construction summary:

- airport-month route shocks are mapped to NUTS2 using the airport crosswalk
- monthly regional shocks are aggregated to quarter
- OECD regional quarterly RHPI (`RHPI`, `FREQ=Q`) is merged with route-shock intensities
- event-study design uses the **first qualifying persistent opening/closure event per region** with a `[-8, +8]` window (omitting `t=-1`)
- TWFE estimation includes region FE and quarter FE with two-way clustered SE (region and time)
- joint Wald tests are run for pre-trend leads (`t=-8` to `t=-2`)

Output files:

- `data/processed/panel_nuts2_quarterly_route_shocks.parquet`
- `results/event_study_pretrend_coefficients.csv`
- `results/event_study_pretrend_summary.json`
- `results/event_study_pretrend_summaries.txt`

## Advanced FE / IV / Event-Study Results (Preliminary, With Diagnostics)

From `results/advanced_model_coefficients.csv` and `results/advanced_model_summaries.txt`:

- **Country-year IV + country/year FE** (`n=157`): lagged net migration rate becomes statistically insignificant in the IV specification (`coef ≈ -0.066`, `p≈0.80`).
- **Asylum-IV variant (cleaner exclusion story, single instrument)** (`n=157`): lagged net migration rate remains imprecisely estimated (`coef ≈ 0.198`, `p≈0.741`).
- **IV first-stage strength is modest / weak-ish**:
  - partial F-stat ≈ `5.65` (clustered, 2 instruments)
  - this is below common strong-instrument rules of thumb
- **Overidentification tests reject** in this run (Sargan/Basmann p-values < 0.01), which is a warning sign that at least one instrument component may violate exclusion.
- **Quarterly route-shock FE models**:
  - `L1_air_yoy` remains positive and significant (`~0.04`, p around `0.01`)
  - route opening/closure aggregate shock terms are not yet significant at the country-quarter level
- **NUTS2 regional TWFE**:
  - `L1_air_growth` is negative/significant in the baseline regional FE model (interpret cautiously; likely composition/supply-constraint heterogeneity)
  - post-2020 interaction suggests migration-price association changed materially in the pandemic/recovery period
- **Regional persistent-route event studies (TWFE pre-trend tests)**:
  - opening events: lead-joint test `chi2(7) ≈ 11.27`, `p≈0.127` (do not reject pre-trends)
  - closure events: lead-joint test `chi2(7) ≈ 11.61`, `p≈0.114` (do not reject pre-trends)
  - event-time effects are currently imprecise, but diagnostics support continuing the design

Interpretation:

- The advanced pipeline is now strong enough to diagnose identification problems, not just estimate associations.
- The airport-to-region mapping and first event-study diagnostics are now in place.
- The next upgrade should prioritize **instrument quality/validity improvements** and **more exogenous route-shock tagging** (airline/slot/regulatory events).

## Current Approaches in the Literature (What Journal Papers Typically Do)

### Migration -> Housing Prices

Common approaches:

- **Two-way FE panels** with macro controls and lag structures (baseline benchmark).
- **Shift-share / Bartik-style IVs** using pre-period migrant settlement shares interacted with origin-country shocks.
- **Spatial models** to account for spillovers across neighboring housing markets.
- **Event studies / policy shocks** (e.g., refugee reallocations, visa reforms, sudden inflow shocks).

Relevant references (methods anchors):

- Saiz (2007, JUE / NBER WP) documents immigration impacts on rents and housing values and emphasizes supply elasticity differences.
- Degen & Fischer (2017) estimate immigration effects on Swiss house prices/rents using regional variation and instrumental-variable logic.
- Helfer, Hlavac & Schmidpeter (2023, *Regional Science and Urban Economics*) decompose migration effects on house prices and highlight heterogeneous impacts.
- Zhu & Pryce (2025, accepted manuscript) use a spatial panel IV design for England and Wales and explicitly model spatial diffusion.

### Air Connectivity / Mobility -> Real Estate

Common approaches:

- **Hedonic price models** around airports (noise/accessibility capitalization into house prices).
- **Difference-in-differences / event studies** around route openings/closures or airport shocks.
- **Urban-growth quasi-experiments** using exogenous changes in air service to identify broader economic effects.
- **Tourism/housing panels** (especially for short-term demand pressure channels) using dynamic panel and cross-sectional heterogeneity.

Relevant references (methods anchors):

- Blonigen & Cristea (2015, *Journal of Urban Economics*) use a quasi-natural experiment to identify air service effects on urban growth.
- Recent tourism-house price work (e.g., Tomal, 2025) uses panel methods to link mobility-driven demand to housing prices and highlights regional heterogeneity.

## Recommended Journal-Quality Econometric Strategy (Next Phase)

### Paper Framing

A strong paper should not only ask whether mobility correlates with house prices, but **which mobility channel matters**:

- permanent population inflows (migration),
- temporary mobility / accessibility (air traffic),
- or both, with different lag structures and market tightness effects.

### Preferred Empirical Design (Country-Level Start, Regional Upgrade)

1. **Country-level baseline (already built)**

- Two-way FE annual and quarterly panels
- Distributed lags for migration and flight growth
- Pandemic-period interactions (`2020-2022`) and post-2022 recovery heterogeneity

2. **Migration causal design (priority)**

- Construct a **shift-share IV**:
  - pre-period migrant shares by origin-country in each destination
  - interacted with origin-specific push shocks (conflict, GDP shocks, asylum waves, etc.)
- Estimate 2SLS with country and year FE
- Test overidentification / weak-IV diagnostics (now implemented in the first advanced IV pass; current diagnostics indicate improvement is needed)

3. **Flight causal design (priority)**

- Use exogenous route/connectivity shocks rather than aggregate passenger totals alone:
  - low-cost carrier entry/exit,
  - regulatory or slot allocation changes,
  - airport closures/strikes/natural disruptions,
  - route network shocks affecting accessibility but plausibly not house prices directly (except through demand)
- Implement event-study / DiD around identified shocks
- Monthly airport-partner route shock inputs, persistent event definitions, and airport-to-region mapping are now implemented
- Next step is to tag events by exogenous source (slot/regulatory/airline-network shocks) and strengthen exclusion restrictions

4. **Regional panel extension (high-value upgrade)**

- Move from country to `NUTS2` (or metro/airport catchment) if feasible:
  - regional air passenger data / airport-linked regions
  - regional house-price proxies or transaction indices
  - regional migration/population and labor-market controls
- This materially improves identification and publication potential because national panels hide within-country reallocation.
- A first NUTS2 annual panel is now implemented (OECD RHPI + Eurostat regional mobility/macros), enabling immediate regional FE testing.

5. **Spatial econometrics**

- Spatial lag / spatial Durbin panel models or Conley-type robust inference
- Important because migration and airport access shocks spill over into nearby markets

### Robustness Package (Expected in Review)

- Alternative dependent variables:
  - nominal HPI growth
  - deflated HPI growth
  - level changes vs log changes
- Excluding `2020-2021` pandemic shock years
- Pre-trend checks in event studies
- First-pass pre-trend checks are implemented for persistent route opening/closure event studies (regional quarterly panel)
- Heterogeneity:
  - supply elasticity / construction responsiveness
  - tourism intensity
  - mortgage-rate regime / euro vs non-euro countries
  - high-demand capitals vs non-capitals

## Process / Reproducibility

Run order:

```bash
python3 scripts/fetch_eurostat.py
python3 scripts/build_panels.py
python3 scripts/estimate_models.py
python3 scripts/fetch_advanced_sources.py
python3 scripts/build_shiftshare_iv.py
python3 scripts/build_flight_shocks.py
python3 scripts/build_airport_nuts2_crosswalk.py
python3 scripts/build_regional_panel.py
python3 scripts/build_regional_route_quarterly_panel.py
python3 scripts/estimate_event_study_pretrends.py
python3 scripts/estimate_advanced_models.py
```

Generated outputs:

- `data/raw/`: Eurostat pulls (CSV)
- `data/processed/panel_annual.parquet`
- `data/processed/panel_quarterly.parquet`
- `results/model_coefficients.csv`
- `results/model_summaries.txt`
- `results/sample_stats.csv`
- `data/processed/panel_annual_iv.parquet`
- `data/processed/panel_quarterly_airport_shocks.parquet`
- `data/processed/panel_nuts2_annual.parquet`
- `data/processed/panel_nuts2_quarterly_route_shocks.parquet`
- `data/metadata/airport_nuts2_crosswalk.csv`
- `data/metadata/airport_nuts2_crosswalk_summary.json`
- `results/advanced_model_coefficients.csv`
- `results/advanced_model_summaries.txt`
- `results/event_study_pretrend_coefficients.csv`
- `results/event_study_pretrend_summary.json`

## Source Links (Data + Methods)

Eurostat / official data docs:

- Eurostat data browser (HPI annual): `https://ec.europa.eu/eurostat/databrowser/view/PRC_HPI_A/default/table?lang=en`
- Eurostat data browser (HPI quarterly): `https://ec.europa.eu/eurostat/databrowser/view/PRC_HPI_Q/default/table?lang=en`
- Eurostat data browser (air passengers, AVIA_PAOC): `https://ec.europa.eu/eurostat/databrowser/view/AVIA_PAOC/default/table?lang=en`
- Eurostat data browser (net migration rate, TPS00019): `https://ec.europa.eu/eurostat/databrowser/view/TPS00019/default/table?lang=en`
- Eurostat data browser (immigration, TPS00176): `https://ec.europa.eu/eurostat/databrowser/view/TPS00176/default/table?lang=en`
- Eurostat data browser (emigration, TPS00177): `https://ec.europa.eu/eurostat/databrowser/view/TPS00177/default/table?lang=en`
- Eurostat Statistics Explained: House price statistics: `https://ec.europa.eu/eurostat/statistics-explained/index.php?title=House_price_statistics_-_house_price_index`
- Eurostat Statistics Explained: Air passenger transport statistics: `https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Air_passenger_transport_statistics`
- Eurostat Statistics Explained: Migration and migrant population statistics: `https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Migration_and_migrant_population_statistics`

OECD / World Bank (newly used for advanced pipeline):

- OECD SDMX API docs: `https://www.oecd.org/en/data/insights/data-explainers/2024/09/api.html`
- OECD regional/national house prices dataflow (`DSD_RHPI@DF_RHPI_ALL`): `https://sdmx.oecd.org/public/rest/dataflow/OECD.SDD.TPS/DSD_RHPI@DF_RHPI_ALL/1.0`
- OECD RHPI data endpoint (CSV API pattern used): `https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_RHPI@DF_RHPI_ALL,1.0/.?startPeriod=1990&format=csvfile`
- World Bank Indicators API docs: `https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation`

Spatial mapping / geodata sources (newly used):

- OurAirports airport catalog (CSV): `https://ourairports.com/data/airports.csv`
- Eurostat GISCO NUTS GeoJSON distribution: `https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/`

Literature / methods references (used to shape model strategy):

- Saiz (NBER working paper page): `https://www.nber.org/papers/w14188`
- Degen & Fischer (2017): `https://link.springer.com/article/10.1007/BF03399367`
- Helfer, Hlavac & Schmidpeter (2023): `https://www.sciencedirect.com/science/article/pii/S0166046223000186`
- Zhu & Pryce (2025 accepted manuscript record): `https://eprints.whiterose.ac.uk/id/eprint/221511/`
- Blonigen & Cristea (2015, JUE record): `https://ideas.repec.org/a/eee/juecon/v86y2015icp128-146.html`
- Tomal (2025): `https://link.springer.com/article/10.1007/s11146-025-10108-x`

## Immediate Next Research Upgrades

1. Strengthen the migration IV (refugee/asylum-focused push shocks, alternative base-share windows, and cleaner exclusion tests).
2. Build exogenous route-shock tags (slot policy changes, regulatory shocks, airline entry/exit episodes) on top of the persistent-event panel.
3. Extend airport exposure mapping beyond administrative NUTS2 assignment (catchment areas / travel-time weights / multi-airport metro exposure).
