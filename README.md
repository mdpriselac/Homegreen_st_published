# Green Coffee in the USA (Streamlit site)

Public Streamlit app for the green-coffee market database. This README is a
maintainer guide to the **Analytics** section: how its data flows, what each
analysis computes, and how to test and regenerate it. Run the app with
`streamlit run app.py`.

## Data flow

```
pipeline repo ("Full App Testing copy"): scrape -> ingest -> clean -> sync
        |
        v
Supabase (coffees, coffee_attributes, sellers)
        |
        v   analytics/db_access/coffee_data_extractor.py
per-coffee frame (one row per cleaned coffee)
        |       uses analytics/processing/data_hygiene.py and varietals.py
        v
analytics/processing/*   (distinctiveness, price, interaction, cross-feature,
        |                  turnover, co-occurrence; shared stats in stat_utils.py)
        v   analytics/frontend/data_cache_generator.py
analytics/data/frontend_cache/frontend_cache.json   (single file; the only one the loader reads)
        |
        v   analytics/frontend/cached_data_loader.py
page_apps/analytics/*   (tabs) via page_apps/analytics_page.py
```

The pages never compute analyses and never query the analytics tables: they
only read the cache.

**Extractor.** Pulls `coffees` + `coffee_attributes` (cleaned rows) with the
anon key and builds the per-coffee frame (country, subregion/region key,
process, varietals, flavors with family/genus/species, `price_per_lb`,
`first_observed`/`last_observed`, `is_active`, seller) and derived formats
(contingency, TF-IDF documents, hierarchical, `cross_feature_df`).

**Hygiene (shared, `data_hygiene.py` / `varietals.py`).**
- Placeholders (`UNKNOWN`, `Unknown`, `""`, `N/A`, null, ...) are *missing*,
  never categories (`PLACEHOLDER_STRINGS`). Flavors with a falsy
  family/genus/species are dropped at that level.
- A region is always `"<Country>_<Subregion>"` (`make_region_key`); a missing
  country or subregion gives no region. Known spelling variants are merged
  first (`SUBREGION_ALIASES`, e.g. "Sao Paolo" -> "Sao Paulo"; the data is plain
  ASCII).
- Process is mapped by exact lookup (`PROCESS_MAP`); unrecognised values become
  missing with a logged warning. Monsoon/Decaf map to "Other".
- Varietals are canonicalised by `normalise_varietal` (SL-28/SL 28 -> SL28,
  accented/unaccented Catuai -> Catuai, Gesha -> Geisha, ...). Distinct
  cultivars (Yellow Catuai vs Catuai, Pink Bourbon vs Bourbon) stay separate.
  `expand_varietals(df, mode)` is the only expansion routine: `"weighted"`
  (each coffee's weights sum to 1; descriptive counts/shares only) or
  `"single"` (only single-varietal coffees; required for every statistical test).
- Headline counts come from one function (`headline_counts`); regions are
  country+subregion keys.

**The cache is generated offline, never in the app.** The pipeline repo's
`run_pipeline.sh` has a `cache` step (after a successful sync) that runs
`generate_cache_standalone.py` in this repo, using this repo's `.venv` and
`.streamlit/secrets.toml` (Supabase anon key), and writes
`analytics/data/frontend_cache/`. It never commits or pushes. **Committing and
pushing the regenerated cache is a manual step**; the deployed app reads the
committed files.

## Methodology

Significance everywhere is Benjamini-Hochberg q < 0.05 (`stat_utils.apply_bh`),
applied within a natural family of tests; raw `p_value` is kept next to
`q_value` for display/debugging.

### Distinctiveness (`processing/distinctiveness.py`, cache builders in `distinctiveness_cache.py`)
Unit of analysis is the **coffee**. For a unit (country / region / seller),
taxonomy level (family / genus / species) and flavor, a one-sided Fisher exact
test (over-representation) compares coffees in the unit that list the flavor
with all other coffees. BH is applied within each unit type x level family.
A flavor is *distinctive* if q < `Q_THRESHOLD` **and** lift (share in unit /
share elsewhere) >= `MIN_LIFT`. Only units with >= `MIN_UNIT_COFFEES` coffees and
flavors in >= `MIN_FLAVOR_COFFEES` of them are tested; all tested rows are kept.
- **Seller-support rule** (country and region units): the flavor's coffees must
  come from >= `MIN_SELLERS` sellers and no seller may supply more than
  `MAX_SELLER_SHARE` of them, so one seller's vocabulary cannot pose as an
  origin trait. It does not change p or q.
- The catch-all family "Other" is excluded (`EXCLUDED_FLAVORS`).
- Unit signatures are the top `TOP_N_SIGNATURE` distinctive flavors (empty if
  none; never padded). JSD (vs the rest of the market) and diversity (entropy)
  use **per-coffee-normalised** distributions (each coffee contributes weight 1
  split across its distinct flavors) shrunk toward market shares with `ALPHA`
  pseudo-coffees, so small units and verbose sellers are not inflated.
- Coffees with a missing unit value are excluded from that unit type's
  analysis, including its baseline.

### Price (`processing/price_analysis.py`)
Price per coffee is `cheapest_per_lb` (the cheapest per-lb price offered,
usually the largest bag); `average_per_lb` mixes small-bag and bulk prices and
is not used. Group premium = group median minus overall median. Each group is
tested vs all other coffees (Mann-Whitney) with BH across the groups of a
category; flavors (with vs without) are BH-corrected per taxonomy level.
Category-level Kruskal-Wallis reports eta-squared based on H (`eta2_h`).
Varietal prices use single-varietal coffees only.

### Interactions (`processing/interaction_analysis.py`)
Price interaction premium = median(A and B) - (median(A) + median(B) -
median(all)), i.e. departure from additivity. It is tested by a seeded
permutation test (`InteractionAnalyzer.N_PERMUTATIONS`): B membership is
permuted within A and within not-A, which keeps all marginal counts; BH per
interaction type. Flavor interactions compare observed vs independence-expected
rates with an exact binomial test (integer counts), BH per interaction type over
everything tested. All use single-varietal coffees.

### Cross-feature associations (`processing/cross_feature_analysis.py`)
Categorical pairs: chi-square + Cramer's V behind a **Cochran guard**
(`stat_utils.validated_chi_square`): if expected counts are too sparse, rows and
columns with total < 10 are collapsed once into "Other"; if still invalid a
seeded Monte Carlo p-value with fixed margins is used (`status` is
`ok` | `collapsed` | `monte_carlo`; `insufficient` only for < 2 rows/cols or n < 30).
Category vs price: Kruskal-Wallis with eta-squared H. Each association row has
`effect_size_metric` (`cramers_v`, `eta2_h`, `mean_cramers_v`); the metrics are
not comparable and should be charted separately. Tests use single-varietal
coffees; descriptive heatmaps use weighted expansion (fractional counts).

### Turnover (`processing/turnover_analysis.py`)
Data facts (constants in the module): expiry tracking was unreliable before
`TRACKING_START` (2025-06-18), so coffees that expired before it are excluded;
coffees first seen on or before `INITIAL_INVENTORY_CUTOFF` (2024-04-05) are the
starting inventory (left-censored) and are excluded, as is the first scrape
month from monthly new listings. `last_observed` is the last scrape the coffee
was seen listed. This is survival analysis: active coffees are right-censored,
medians are Kaplan-Meier, groups are compared by log-rank (group vs rest) with
BH across groups. **Duration definition:** each sighting stands for one scrape
interval I (median gap between scrapes): expired = (last - first) + I, active
(censored) = (last - first) + I/2. Sellers with no listing seen in the last
`SELLER_INACTIVE_DAYS` (60) days are treated as closed and excluded from
turnover rates (listed in the results). The observation window and exclusion
counts are in the cache (`observation_window`) so the page can state them.

### Co-occurrence (`processing/cooccurrence_analysis.py`)
Over coffees with flavor notes: P(B | A) with the base rate P(B) and ratio
P(B|A)/P(B) for each flavor (pairs need >= `MIN_PAIR_COUNT` coffees; top
`TOP_N_PER_FLAVOR` kept). "Avoiding" pairs (appear together less than chance,
including pairs that never co-occur) use a one-sided Fisher exact test
(`less`), BH over all pairs tested, for flavors in >= `MIN_FLAVOR_SUPPORT`
coffees (`AVOID_Q_THRESHOLD`).

## Constants

Tunables are module-level constants at the top of each module (e.g.
`distinctiveness.MIN_UNIT_COFFEES`, `distinctiveness.MAX_SELLER_SHARE`,
`cooccurrence_analysis.MIN_PAIR_COUNT`, `turnover_analysis.TRACKING_START`).
Shared statistics are in `stat_utils.py`; shared cleaning in `data_hygiene.py`
and `varietals.py`. Check the module rather than this README for current values.

## Running tests and regenerating the cache

```bash
.venv/bin/python -m pytest tests          # from the repo root

# regenerate the cache locally (reads Supabase via .streamlit/secrets.toml;
# takes a while; writes analytics/data/frontend_cache/)
.venv/bin/python generate_cache_standalone.py
```

Preview a cache generated elsewhere with `ANALYTICS_CACHE_DIR=/absolute/dir streamlit run app.py`
(the loader reads `<dir>/frontend_cache.json`; `ANALYTICS_SHOW_TURNOVER=0` hides the Turnover tab and lifespan rankings).

Normally the pipeline's `run_pipeline.sh` does this after a sync. Review the
diff of the cache and commit/push it manually.

## Known caveats

- **Seller vocabulary confounding:** sellers describe coffees in their own
  words and sell particular origins, so origin "distinctiveness" can partly be
  a seller effect. The seller-support rule mitigates but does not remove it.
- **Small groups:** units near the minimum sizes give wide intervals; most
  distinctive flavors at species level rest on a handful of coffees. Sparse
  chi-square tables use collapsing / Monte Carlo (p floor ~ 1/2001).
- **Compound subregions** (comma-separated values such as
  "Cerrado, Sao Paolo", ~60 coffees) are left as-is and form their own regions.
- **Varietal data is messy:** free text, percentages and non-varietals in the
  source; tests are restricted to single-varietal coffees (about half of the coffees that list a varietal).
- **Turnover:** Kaplan-Meier medians are undefined when under half of a group's
  coffees expired; very short lifespans can reflect scraper artefacts (a
  Shopify URL-parameter issue affected one seller).
- **iCloud:** if the project lives in iCloud, the first Python import (and venv
  files) can be very slow while files download; keep the venv in a `.nosync`
  location or pinned locally, and avoid recursive greps over `analytics/data`.
