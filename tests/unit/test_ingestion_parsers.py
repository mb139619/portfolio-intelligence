"""
Parser tests for the FRED and French ingesters.

We can't hit the live endpoints in CI, but the parsing logic — the fiddly part —
is pure and fully testable against representative sample payloads.
"""

import io
import zipfile

import polars as pl
import pytest

from src.ingestion.rates import _parse_fred_csv
from src.ingestion.french import _parse_french_csv, FrenchIngester


class TestFredParser:
    def test_basic(self):
        csv = "observation_date,DGS10\n2015-01-02,2.12\n2015-01-05,2.04\n"
        df = _parse_fred_csv(csv)
        assert df.columns == ["date", "value"]
        assert df.height == 2
        assert df["value"][0] == pytest.approx(2.12)

    def test_missing_values_dropped(self):
        csv = "observation_date,DGS10\n2015-01-02,2.12\n2015-01-03,.\n2015-01-05,2.04\n"
        df = _parse_fred_csv(csv)
        # The '.' row is dropped
        assert df.height == 2

    def test_alternate_date_header(self):
        csv = "DATE,DFF\n2020-03-15,0.25\n2020-03-16,0.10\n"
        df = _parse_fred_csv(csv)
        assert df.height == 2
        assert df["value"][1] == pytest.approx(0.10)


SAMPLE_FRENCH = """This file was created using the 100% etc.

,Mkt-RF,SMB,HML,RF
20230102, 0.50,-0.20, 0.30, 0.018
20230103,-0.10, 0.05,-0.15, 0.018
20230104, 1.20, 0.40, 0.10, 0.018

  Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
2023, 5.0, 1.0, 2.0, 4.5
"""


class TestFrenchParser:
    def test_parses_daily_block(self):
        df = _parse_french_csv(SAMPLE_FRENCH)
        assert set(df.columns) == {"date", "factor", "value", "source"}
        # 3 daily rows × 4 factors = 12 records
        assert df.height == 12

    def test_excludes_annual_block(self):
        df = _parse_french_csv(SAMPLE_FRENCH)
        # Only 3 distinct daily dates, the annual '2023' row must be excluded
        assert df["date"].n_unique() == 3

    def test_percent_to_decimal(self):
        df = _parse_french_csv(SAMPLE_FRENCH)
        # Mkt-RF on 2023-01-02 was 0.50% -> 0.005
        v = df.filter(
            (pl.col("date") == pl.date(2023, 1, 2)) & (pl.col("factor") == "Mkt-RF")
        )["value"][0]
        assert v == pytest.approx(0.005)

    def test_factor_names(self):
        df = _parse_french_csv(SAMPLE_FRENCH)
        assert set(df["factor"].unique().to_list()) == {"Mkt-RF", "SMB", "HML", "RF"}

    def test_fetch_parses_zip(self, monkeypatch):
        """Simulate the network: patch the http helper to return a zipped CSV."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("F-F_test.csv", SAMPLE_FRENCH)
        zip_bytes = buf.getvalue()

        class FakeResp:
            content = zip_bytes
            def raise_for_status(self):
                pass

        import src.ingestion.french as fr
        monkeypatch.setattr(fr, "get_with_retry", lambda *a, **k: FakeResp())

        df = FrenchIngester().fetch("FF3", start="2023-01-01")
        assert df.height == 12


class TestRetry:
    def test_retries_then_succeeds(self, monkeypatch):
        """First two attempts time out, third succeeds — should return cleanly."""
        import src.ingestion.http as http

        calls = {"n": 0}

        class FakeResp:
            status_code = 200
            text = "ok"
            def raise_for_status(self):
                pass

        def flaky_get(*a, **k):
            calls["n"] += 1
            if calls["n"] < 3:
                raise http.requests.Timeout("simulated timeout")
            return FakeResp()

        monkeypatch.setattr(http.requests, "get", flaky_get)
        monkeypatch.setattr(http.time, "sleep", lambda s: None)  # no real waiting

        resp = http.get_with_retry("http://example.com", max_retries=4, retry_delay=0.0)
        assert resp.text == "ok"
        assert calls["n"] == 3

    def test_sends_browser_user_agent(self, monkeypatch):
        # The default headers must include a non-python User-Agent (WAF fix)
        import src.ingestion.http as http
        captured = {}

        class FakeResp:
            status_code = 200
            text = "ok"
            def raise_for_status(self):
                pass

        def capture_get(url, params=None, headers=None, timeout=None):
            captured["headers"] = headers
            return FakeResp()

        monkeypatch.setattr(http.requests, "get", capture_get)
        http.get_with_retry("http://example.com")
        ua = captured["headers"]["User-Agent"]
        assert "python-requests" not in ua.lower()
        assert "Mozilla" in ua

    def test_gives_up_after_max(self, monkeypatch):
        import src.ingestion.http as http

        def always_timeout(*a, **k):
            raise http.requests.Timeout("always")

        monkeypatch.setattr(http.requests, "get", always_timeout)
        monkeypatch.setattr(http.time, "sleep", lambda s: None)

        with pytest.raises(RuntimeError, match="failed after"):
            http.get_with_retry("http://example.com", max_retries=3)
