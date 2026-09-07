# Dashboard

A static web report over the analytics. The pipeline is one-directional and
entirely offline:

```
portfolio.json  →  ingestion  →  analytics  →  web/data/analysis.json  →  static SPA
```

Nothing is computed at view time. The analysis runs once, locally, and the
frontend renders a result someone else already produced.

---

## Why it is built this way

The obvious alternative — a Python web service that computes on request — was
rejected for three measured reasons.

**Memory.** Importing numpy, scipy, sklearn, statsmodels, polars, duckdb and
plotly costs ~235 MB of resident memory before a single calculation runs; peak
usage during an HMM fit and figure serialisation is closer to 350-400 MB. That
does not fit comfortably in the 512 MB free tiers, and paying for a server to
recompute a result that changes once a day is poor value.

**Yahoo from a datacenter IP.** `yfinance` is rate-limited far more aggressively
from cloud hosts than from a residential connection. Running ingestion locally
sidesteps the problem entirely, and is the reason the universe can be
*unbounded*: any ticker you can download is a ticker you can chart.

**Ephemeral filesystems.** Free hosts wipe the disk on restart, so a Parquet
data lake written at runtime does not survive. Here there is no runtime.

The consequence is that hosting is genuinely free — a static file server has no
RAM limit, no cold start and nothing to keep awake — and the same output opens
from disk with no server at all.

---

## Usage

```bash
python -m src.export                      # ingest, analyse, write the payload
python -m src.export --serve              # ...then serve it and open a browser
```

`--serve` is the normal way to view it. Opening `web/index.html` directly from
disk does not work: the browser blocks the `fetch()` of the data file under the
`file://` protocol. The page detects this and says so rather than failing
silently.

### Options

| Flag | Effect |
|------|--------|
| `--config PATH` | portfolio specification (default `portfolio.json`) |
| `--out PATH` | where to write the payload (default `web/data/analysis.json`) |
| `--skip-ingest` | reuse the local Parquet store; download nothing |
| `--skip-quality` | skip the post-ingestion data quality report |
| `--only a,b` | rebuild named sections only — useful while iterating |
| `--serve` | serve `web/` over HTTP and open a browser |
| `--port N` | port for `--serve` (default 8000) |
| `--standalone` | also emit a single self-contained HTML |
| `--indent N` | pretty-print the JSON for debugging |

### The standalone build

```bash
python -m src.export --standalone
```

Inlines the payload, the stylesheet, the application and the Plotly bundle into
one ~6 MB HTML file at `web/data/analysis.html`. It opens with a double click,
needs no server, and can be emailed as a single artefact. It is a build output
and is gitignored.

---

## Defining a portfolio

`portfolio.json` is the only input. The universe is whatever you put in it.

```json
{
  "name": "Demo 60/40-ish",
  "start": "2018-01-01",
  "end": null,
  "cov_method": "ledoit_wolf_cc",
  "positions": [
    { "ticker": "SPY", "name": "S&P 500 ETF", "asset_class": "equity", "weight": 0.6 },
    { "ticker": "TLT", "name": "20Y Treasury ETF", "asset_class": "fixed_income", "weight": 0.4 }
  ],
  "factor_start": "1990-01-01",
  "settings": { "regime_states": 2 }
}
```

Weights must sum to 1; `asset_class` drives the outlier bands in the data
quality checks. Validation happens before anything is downloaded, so a typo
fails immediately rather than twenty seconds into a fetch.

**`factor_start` is deliberately much earlier than `start`.** Factor-based
stress replay is the one method that reaches crises predating the holdings
themselves — with factors from 1990 the portfolio can be pushed through the
2008 crisis even though the book starts in 2018. It also gives the regime model
decades rather than years to estimate transition probabilities from. The whole
factor history is one file of a few hundred KB, so the depth is free.

### Settings

| Key | Default | Meaning |
|-----|---------|---------|
| `rolling_window` | 63 | window for rolling %RC and volatility |
| `beta_window` | 252 | window for rolling factor betas |
| `corr_window` | 63 | window for average correlation |
| `n_clusters` | 3 | correlation clusters |
| `pca_method` | `covariance` | or `correlation` |
| `regime_states` | 2 | HMM states |
| `regime_search_reps` | 20 | EM restarts, guards against local optima |
| `var_confidence` | 0.99 | confidence for the tail page |
| `evt_threshold_quantile` | 0.95 | POT threshold |
| `frontier_points` | 50 | portfolios traced on the frontier |
| `cloud_points` | 1500 | random portfolios behind the frontier |
| `hac_lags` | 5 | Newey-West lags; `null` for classical OLS SE |

---

## Payload schema

The SPA knows the schema but nothing about finance. Every section is the same
shape, which is why adding an analysis requires no JavaScript.

```jsonc
{
  "meta": {
    "generated_at": "...",
    "portfolio": { "name": "...", "positions": [...], "cov_method": "..." },
    "coverage":   { "start": "...", "end": "...", "observations": 2149 },
    "warnings":   ["..."],
    "quality":    { "findings": 5, "critical": 0, "detail": [...] }
  },
  "sections": [
    {
      "id": "risk", "title": "...", "subtitle": "...",
      "error":   "",                                    // set if the section failed
      "stats":   [{ "label", "value", "format", "hint", "tone" }],
      "figures": [{ "id", "title", "note", "data", "layout" }],  // Plotly specs
      "tables":  [{ "id", "title", "columns", "rows", "formats", "note" }],
      "notes":   ["..."]
    }
  ]
}
```

Values are raw numbers, never pre-formatted strings; `format` is one of
`percent`, `number`, `ratio`, `integer`, `text` and the frontend decides how to
render it. That keeps the JSON usable by anything else that wants to read it.

Two encoding details are load-bearing. NaN and infinity are converted to `null`,
because Python's `json.dumps` emits bare `NaN` and `Infinity` tokens which are
not valid JSON and make the browser's `JSON.parse` throw — and both values arise
naturally here (a regime with one observation, an EVT shortfall with GPD shape
ξ ≥ 1). The final serialisation then uses `allow_nan=False` so anything that
slipped through fails loudly at build time instead of silently in a browser.

---

## Adding a section

1. Write `build_<name>(ctx: Context) -> dict` in `src/export/build.py`, returning
   `encode.section(...)`. Pull expensive shared objects off `ctx` — covariance,
   factor model, PCA and regimes are memoised there and reused across sections.
2. Register it in `SECTIONS`.

No frontend change. If the analytic needs a chart that does not exist yet, add
it to `src/viz/plots.py` where it is unit-tested and the notebook can use it
too — not to the export layer.

Sections are individually wrapped: a failure renders as a visible error panel on
that page and the other seven build normally. Free data is patchy and the
Fama-French library publishes with a lag, so a book with eight months of history
genuinely cannot support a 252-day rolling beta. Saying so is better than
quietly dropping a risk panel.

---

## Publishing

`web/` is a static directory. Any static host serves it, including GitHub Pages,
which needs no build step and does not sleep:

1. Commit `web/` — `analysis.json` and the vendored Plotly bundle are tracked on
   purpose, because they are what the host serves.
2. Point Pages at the branch and the `/web` folder.

Re-publishing is `python -m src.export` followed by a commit. The payload is
~1.4 MB and the Plotly bundle ~4.8 MB, both served gzipped.

---

## Cost of a build

Roughly 15-35 seconds end to end on the demo universe, dominated by the regime
model (a 2-state Markov-switching fit with 20 EM restarts over ~9,000 daily
market observations) and the rolling factor betas. Ingestion adds a few seconds
and is incremental, so re-running is cheap. `--skip-ingest` removes it entirely
and `--only` narrows the rebuild while iterating on one page.
