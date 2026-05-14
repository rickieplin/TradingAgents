"""Static registration checks for the ``a_stock`` data vendor.

These tests verify that the porting of the A-stock fork wired
``a_stock`` into ``interface.py``'s vendor registry correctly: every
expected method appears with an ``a_stock`` implementation, the new
``signal_data`` category is registered, and ``route_to_vendor`` picks
the right impl when the config asks for ``a_stock``.

No network is touched. The a_stock impls keep their network calls
inside function bodies (``import mootdx`` / ``import akshare`` etc are
lazy), so simply importing the module never hits a broker server.
"""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config


# The 9 methods shared across vendors + 8 A-stock-only signal methods.
_SHARED_METHODS = (
    "get_stock_data",
    "get_indicators",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
)

_SIGNAL_METHODS = (
    "get_profit_forecast",
    "get_hot_stocks",
    "get_northbound_flow",
    "get_concept_blocks",
    "get_fund_flow",
    "get_dragon_tiger_board",
    "get_lockup_expiry",
    "get_industry_comparison",
)


@pytest.mark.unit
class AStockVendorRegistrationTests(unittest.TestCase):
    """``a_stock`` is registered alongside yfinance / alpha_vantage."""

    def test_vendor_appears_in_vendor_list(self):
        self.assertIn("a_stock", interface.VENDOR_LIST)

    def test_all_shared_methods_have_a_stock_impl(self):
        for method in _SHARED_METHODS:
            with self.subTest(method=method):
                self.assertIn(method, interface.VENDOR_METHODS)
                self.assertIn("a_stock", interface.VENDOR_METHODS[method])
                self.assertTrue(callable(interface.VENDOR_METHODS[method]["a_stock"]))

    def test_yfinance_still_default_for_shared_methods(self):
        # We deliberately did NOT flip defaults to a_stock — this is the
        # opt-in invariant. yfinance must still be present so US runs work.
        for method in _SHARED_METHODS:
            with self.subTest(method=method):
                self.assertIn("yfinance", interface.VENDOR_METHODS[method])

    def test_signal_data_category_is_registered(self):
        self.assertIn("signal_data", interface.TOOLS_CATEGORIES)
        registered = set(interface.TOOLS_CATEGORIES["signal_data"]["tools"])
        self.assertEqual(registered, set(_SIGNAL_METHODS))

    def test_signal_methods_route_only_to_a_stock(self):
        # Signal data has no US/global equivalent — only a_stock implements.
        for method in _SIGNAL_METHODS:
            with self.subTest(method=method):
                vendors = set(interface.VENDOR_METHODS[method].keys())
                self.assertEqual(vendors, {"a_stock"})

    def test_get_category_for_method_finds_signal_data_tools(self):
        for method in _SIGNAL_METHODS:
            with self.subTest(method=method):
                self.assertEqual(
                    interface.get_category_for_method(method),
                    "signal_data",
                )


@pytest.mark.unit
class AStockRoutingTests(unittest.TestCase):
    """``route_to_vendor`` picks the right impl based on config."""

    def setUp(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def tearDown(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_default_config_keeps_yfinance_for_core_categories(self):
        # Regression guard: porting must not flip the US defaults.
        cfg = default_config.DEFAULT_CONFIG
        self.assertEqual(cfg["data_vendors"]["core_stock_apis"], "yfinance")
        self.assertEqual(cfg["data_vendors"]["technical_indicators"], "yfinance")
        self.assertEqual(cfg["data_vendors"]["fundamental_data"], "yfinance")
        self.assertEqual(cfg["data_vendors"]["news_data"], "yfinance")

    def test_signal_data_defaults_to_a_stock(self):
        # signal_data is the only category where a_stock is the only impl, so
        # the default must point at it for the routing to actually work.
        self.assertEqual(
            default_config.DEFAULT_CONFIG["data_vendors"]["signal_data"],
            "a_stock",
        )

    def test_opting_in_routes_get_stock_data_to_a_stock(self):
        set_config({"data_vendors": {"core_stock_apis": "a_stock"}})
        # Stub the a_stock impl so we never touch mootdx.
        sentinel = object()
        with patch.dict(
            interface.VENDOR_METHODS["get_stock_data"],
            {"a_stock": lambda *a, **kw: sentinel},
        ):
            result = interface.route_to_vendor(
                "get_stock_data", "600519", "2026-05-01", "2026-05-10"
            )
        self.assertIs(result, sentinel)

    def test_signal_tool_routes_to_a_stock_under_default_config(self):
        sentinel = object()
        with patch.dict(
            interface.VENDOR_METHODS["get_hot_stocks"],
            {"a_stock": lambda *a, **kw: sentinel},
        ):
            result = interface.route_to_vendor("get_hot_stocks", "2026-05-12")
        self.assertIs(result, sentinel)


@pytest.mark.unit
class AStockLazyImportTests(unittest.TestCase):
    """The a_stock module must not import network libraries at module load.

    Importing ``tradingagents.dataflows.a_stock`` should not require
    ``mootdx``/``akshare``/``requests`` to be installed — those are
    optional extras and must stay inside function bodies so US flows that
    never call into the vendor don't break on bare installs.
    """

    def test_module_imports_without_optional_extras(self):
        # If this import succeeds, the module's top-level imports are clean.
        # (pandas/pd are already main deps via yfinance.)
        from tradingagents.dataflows import a_stock  # noqa: F401

    def test_top_level_imports_do_not_include_optional_extras(self):
        import ast

        from tradingagents.dataflows import a_stock as a_stock_mod

        with open(a_stock_mod.__file__) as f:
            tree = ast.parse(f.read())

        forbidden_at_top_level = {"mootdx", "akshare", "requests", "stockstats"}
        top_level_modules = set()
        for node in tree.body:  # only top-level statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_modules.add(node.module.split(".")[0])

        leaked = forbidden_at_top_level & top_level_modules
        self.assertEqual(
            leaked, set(),
            f"a_stock.py leaks optional deps at top level: {leaked}",
        )
