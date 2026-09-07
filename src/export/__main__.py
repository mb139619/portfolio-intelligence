"""
Build the dashboard payload.

    python -m src.export                     # ingest, analyse, write the payload
    python -m src.export --serve             # ...then serve the SPA in a browser
    python -m src.export --skip-ingest       # reuse whatever is already in the store
    python -m src.export --config other.json --out dist/other.json
    python -m src.export --only overview,risk # rebuild two sections while iterating
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from loguru import logger

from src.config import settings
from src.data_quality.report import run_quality_report
from src.export.build import SECTIONS, build_payload
from src.export.encode import dumps
from src.export.spec import PortfolioSpec
from src.ingestion.pipeline import IngestionPipeline
from src.store.parquet_store import ParquetStore

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
DEFAULT_OUT = WEB_DIR / "data" / "analysis.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.export",
        description="Run the portfolio analysis and write the dashboard payload.",
    )
    p.add_argument("--config", default=str(ROOT / "portfolio.json"),
                   help="portfolio specification (default: portfolio.json)")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"output JSON path (default: {DEFAULT_OUT.relative_to(ROOT)})")
    p.add_argument("--skip-ingest", action="store_true",
                   help="do not download anything; use the local Parquet store as-is")
    p.add_argument("--skip-quality", action="store_true",
                   help="skip the post-ingestion data quality report")
    p.add_argument("--only", default="",
                   help="comma-separated section ids to build "
                        f"({', '.join(s[0] for s in SECTIONS)})")
    p.add_argument("--serve", action="store_true",
                   help="serve the SPA over HTTP and open it in a browser")
    p.add_argument("--port", type=int, default=8000, help="port for --serve")
    p.add_argument("--standalone", action="store_true",
                   help="also write a single self-contained HTML with the data inlined")
    p.add_argument("--indent", type=int, default=None,
                   help="pretty-print the JSON (larger file; useful for debugging)")
    return p.parse_args(argv)


def ingest(spec: PortfolioSpec, store: ParquetStore) -> None:
    pipeline = IngestionPipeline(store)

    logger.info(f"Ingesting prices for {len(spec.tickers)} tickers...")
    failures = []
    for res in pipeline.update_prices(spec.tickers, start=spec.start):
        if not res.success:
            failures.append(f"{res.identifier}: {res.error}")
        logger.info(f"  {res}")

    if failures:
        # A price failure is fatal for that ticker, but the store may still hold
        # earlier history, so let the build decide rather than aborting here.
        logger.warning(
            f"{len(failures)} price download(s) failed:\n  "
            + "\n  ".join(failures)
            + "\nYahoo rate-limits aggressively; retrying usually clears it."
        )

    logger.info("Ingesting rates...")
    for res in pipeline.update_rates(spec.rates, start=spec.start):
        logger.info(f"  {res}")

    logger.info(f"Ingesting factors from {spec.factor_start} (deep history)...")
    for res in pipeline.update_factors(spec.factor_datasets, start=spec.factor_start):
        logger.info(f"  {res}")


def write_standalone(payload_json: str, out_html: Path) -> None:
    """
    Inline the payload and the Plotly bundle into one HTML file.

    Useful because the normal SPA fetches its JSON, which the browser blocks
    under file:// for cross-origin reasons — the standalone build has nothing to
    fetch, so it opens with a double click and can be emailed as one artefact.
    """
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise FileNotFoundError(f"SPA not found at {index}")

    html = index.read_text(encoding="utf-8")
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    app_js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    plotly_js = (WEB_DIR / "vendor" / "plotly.min.js").read_text(encoding="utf-8")

    html = html.replace(
        '<link rel="stylesheet" href="styles.css">',
        f"<style>\n{css}\n</style>",
    )
    html = html.replace(
        '<script src="vendor/plotly.min.js"></script>',
        f"<script>\n{plotly_js}\n</script>",
    )
    # The closing tag is split so a </script> inside the JSON cannot end the block.
    html = html.replace(
        '<script src="app.js"></script>',
        '<script id="payload" type="application/json">'
        + payload_json.replace("</", "<\\/")
        + "</script>\n<script>\n"
        + app_js
        + "\n</script>",
    )
    out_html.write_text(html, encoding="utf-8")


def serve(port: int) -> None:
    import http.server
    import socketserver
    import threading
    import webbrowser

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(WEB_DIR), **kw)

        def log_message(self, fmt, *args):        # quieter than the default
            logger.debug(fmt % args)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        logger.info(f"Serving {WEB_DIR} at {url}  (Ctrl-C to stop)")
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Stopped.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        spec = PortfolioSpec.load(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 2

    logger.info(spec.describe())
    settings.ensure_dirs()
    store = ParquetStore(settings.data_dir)

    if not args.skip_ingest:
        ingest(spec, store)
    else:
        logger.info("Skipping ingestion (--skip-ingest)")

    quality = None
    if not args.skip_quality:
        quality = run_quality_report(
            store, tickers=spec.tickers, asset_classes=spec.asset_classes
        )
        logger.info(
            f"Data quality: {len(quality.findings)} findings "
            f"({quality.n_critical} critical) across {len(quality.tickers)} tickers"
        )
        for f in quality.findings:
            logger.info(f"  {f}")

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    if only:
        known = {s[0] for s in SECTIONS}
        unknown = set(only) - known
        if unknown:
            logger.error(
                f"Unknown section(s): {sorted(unknown)}. Valid: {sorted(known)}"
            )
            return 2

    t0 = time.perf_counter()
    try:
        payload = build_payload(spec, store, quality=quality, only=only)
    except ValueError as e:
        logger.error(str(e))
        return 1

    payload_json = dumps(payload, indent=args.indent)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload_json, encoding="utf-8")

    size_mb = len(payload_json.encode("utf-8")) / 1e6
    failed = [s["id"] for s in payload["sections"] if s.get("error")]
    logger.success(
        f"Built {len(payload['sections'])} sections in "
        f"{time.perf_counter() - t0:.1f}s → {out}  ({size_mb:.2f} MB)"
    )
    if failed:
        logger.warning(f"Sections with errors: {', '.join(failed)}")

    # Keep the served copy in sync when writing elsewhere, so --out and --serve
    # can be combined without the SPA silently showing a stale payload.
    if args.serve and out.resolve() != DEFAULT_OUT.resolve():
        DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, DEFAULT_OUT)

    if args.standalone:
        html_out = out.with_suffix(".html")
        write_standalone(payload_json, html_out)
        logger.success(f"Standalone dashboard → {html_out}")

    if args.serve:
        serve(args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
