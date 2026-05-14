"""Benchmark resolution for A-share suffixes routes to the CSI 300.

The alpha-vs-benchmark calculation in :class:`TradingAgentsGraph` uses
``_resolve_benchmark`` to pick the index a ticker is measured against.
After porting the A-stock vendor we added ``.SH`` / ``.SS`` / ``.SZ`` ->
``000300.SS`` (CSI 300, 沪深300) and ``.BJ`` -> ``899050.BJ`` (北证 50)
entries to ``benchmark_map``. These tests pin that mapping and verify the
existing US / non-US entries still resolve correctly.

Driven directly against ``benchmark_map`` so we don't construct a graph.
"""

from __future__ import annotations

import unittest

import pytest

from tradingagents.default_config import DEFAULT_CONFIG


def _resolve(ticker: str, override: str | None = None) -> str:
    """Replicate ``TradingAgentsGraph._resolve_benchmark`` exactly.

    Keeping the logic copy short and faithful to trading_graph.py so the
    test pins behaviour without requiring an LLM client to instantiate.
    """
    if override:
        return override
    bmap = DEFAULT_CONFIG["benchmark_map"]
    ticker_upper = ticker.upper()
    for suffix, bench in bmap.items():
        if suffix and ticker_upper.endswith(suffix.upper()):
            return bench
    return bmap.get("", "SPY")


@pytest.mark.unit
class AShareBenchmarkMapTests(unittest.TestCase):
    """A-share suffixes route to CSI 300 / BSE 50."""

    def test_shanghai_main_board_routes_to_csi_300(self):
        self.assertEqual(_resolve("600519.SH"), "000300.SS")  # 茅台

    def test_shanghai_star_board_routes_to_csi_300(self):
        # STAR (科创板) tickers can use .SS as the suffix
        self.assertEqual(_resolve("688981.SS"), "000300.SS")  # 中芯国际

    def test_shenzhen_main_board_routes_to_csi_300(self):
        self.assertEqual(_resolve("000858.SZ"), "000300.SS")  # 五粮液

    def test_shenzhen_chinext_routes_to_csi_300(self):
        # ChiNext (创业板) — same suffix as Shenzhen main
        self.assertEqual(_resolve("300750.SZ"), "000300.SS")  # 宁德时代

    def test_beijing_stock_exchange_routes_to_bse_50(self):
        self.assertEqual(_resolve("832000.BJ"), "899050.BJ")

    def test_a_share_routes_are_case_insensitive(self):
        # Suffix matching uppercases the ticker before comparing.
        self.assertEqual(_resolve("600519.sh"), "000300.SS")
        self.assertEqual(_resolve("000858.sz"), "000300.SS")


@pytest.mark.unit
class NonAShareBenchmarkRegressionTests(unittest.TestCase):
    """Existing benchmark entries must keep resolving correctly."""

    def test_us_ticker_with_no_suffix_routes_to_spy(self):
        self.assertEqual(_resolve("NVDA"), "SPY")
        self.assertEqual(_resolve("AAPL"), "SPY")

    def test_us_ticker_with_dotted_class_routes_to_spy(self):
        # BRK.B / BF.B style — no suffix matches the map, fallback to SPY.
        # This is intentional: alpha is computed in USD so SPY is correct.
        self.assertEqual(_resolve("BRK.B"), "SPY")

    def test_japan_suffix_still_routes_to_nikkei(self):
        self.assertEqual(_resolve("7203.T"), "^N225")

    def test_hong_kong_suffix_still_routes_to_hang_seng(self):
        self.assertEqual(_resolve("0700.HK"), "^HSI")

    def test_london_suffix_still_routes_to_ftse(self):
        self.assertEqual(_resolve("HSBA.L"), "^FTSE")

    def test_explicit_override_wins_over_suffix_map(self):
        # The first arg of _resolve_benchmark is config["benchmark_ticker"];
        # when set it short-circuits suffix lookup.
        self.assertEqual(_resolve("600519.SH", override="^GSPC"), "^GSPC")
