"""Graph wiring tests for the A-stock additions.

The fork explicitly documents that adding any analyst means touching 6
interlocking files (analyst body, agent_states, agents/__init__,
conditional_logic, trading_graph, setup). These tests pin the
ports's plumbing so a future edit to any one of those six can't
silently desync from the others — the graph would still compile but the
new node would never fire or its report field would not flow downstream.

We instantiate ``TradingAgentsGraph`` with the ``mock_llm_client``
fixture so no real LLM calls happen.
"""

from __future__ import annotations

import pytest

from tradingagents.default_config import DEFAULT_CONFIG


_A_STOCK_ANALYSTS = ("policy", "hot_money", "lockup")
_LEGACY_ANALYSTS = ("market", "social", "news", "fundamentals")


@pytest.mark.unit
class TestUSFlowRegression:
    """The 4-analyst US flow must still compile after the port."""

    def test_default_4_analyst_graph_compiles(self, mock_llm_client):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        g = TradingAgentsGraph(
            selected_analysts=list(_LEGACY_ANALYSTS),
            config=DEFAULT_CONFIG.copy(),
        )
        nodes = set(g.workflow.nodes.keys())

        # Each legacy analyst should produce 3 nodes: analyst, tools_, Msg Clear
        for at in _LEGACY_ANALYSTS:
            label = at.capitalize()
            assert f"{label} Analyst" in nodes, f"missing {label} Analyst"
            assert f"tools_{at}" in nodes, f"missing tools_{at}"
            assert f"Msg Clear {label}" in nodes, f"missing Msg Clear {label}"

        # The fixed agents (researchers, debaters, managers) must be present.
        for fixed in (
            "Bull Researcher", "Bear Researcher", "Research Manager",
            "Trader", "Aggressive Analyst", "Conservative Analyst",
            "Neutral Analyst", "Portfolio Manager",
        ):
            assert fixed in nodes, f"missing fixed node: {fixed}"

    def test_quality_gate_node_present_for_us_flow(self, mock_llm_client):
        # The Quality Gate is market-agnostic — it runs even for US flows.
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        g = TradingAgentsGraph(
            selected_analysts=list(_LEGACY_ANALYSTS),
            config=DEFAULT_CONFIG.copy(),
        )
        assert "Quality Gate" in g.workflow.nodes


@pytest.mark.unit
class TestAStockAnalystWiring:
    """A-stock analysts are opt-in and wire identically to legacy ones."""

    def test_full_seven_analyst_graph_compiles(self, mock_llm_client):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        g = TradingAgentsGraph(
            selected_analysts=list(_LEGACY_ANALYSTS) + list(_A_STOCK_ANALYSTS),
            config=DEFAULT_CONFIG.copy(),
        )
        nodes = set(g.workflow.nodes.keys())

        # Three new analysts, each with the 3-node pipeline.
        for at in _A_STOCK_ANALYSTS:
            label = at.capitalize()
            assert f"{label} Analyst" in nodes
            assert f"tools_{at}" in nodes
            assert f"Msg Clear {label}" in nodes

    def test_a_stock_analysts_can_run_in_isolation(self, mock_llm_client):
        # If a user selects only A-stock analysts, the graph must still wire
        # cleanly: first analyst -> ... -> Quality Gate -> Bull Researcher.
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        g = TradingAgentsGraph(
            selected_analysts=list(_A_STOCK_ANALYSTS),
            config=DEFAULT_CONFIG.copy(),
        )
        nodes = set(g.workflow.nodes.keys())
        for at in _A_STOCK_ANALYSTS:
            assert f"{at.capitalize()} Analyst" in nodes
        assert "Quality Gate" in nodes
        assert "Bull Researcher" in nodes

    def test_unselected_a_stock_analysts_do_not_appear(self, mock_llm_client):
        # Selecting only legacy analysts must NOT add policy/hot_money/lockup
        # nodes — these are opt-in.
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        g = TradingAgentsGraph(
            selected_analysts=list(_LEGACY_ANALYSTS),
            config=DEFAULT_CONFIG.copy(),
        )
        nodes = set(g.workflow.nodes.keys())
        for at in _A_STOCK_ANALYSTS:
            assert f"{at.capitalize()} Analyst" not in nodes
            assert f"tools_{at}" not in nodes


@pytest.mark.unit
class TestQualityGateInsertion:
    """The Quality Gate sits between the last analyst and Bull Researcher."""

    def test_quality_gate_routes_to_bull_researcher(self, mock_llm_client):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        g = TradingAgentsGraph(
            selected_analysts=["market"],
            config=DEFAULT_CONFIG.copy(),
        )
        # Edges in LangGraph compiled graphs aren't trivially introspectable,
        # but the .workflow (uncompiled StateGraph) exposes them.
        edges = list(g.workflow.edges)
        # Find the edge originating from Quality Gate.
        qg_targets = [tgt for (src, tgt) in edges if src == "Quality Gate"]
        assert "Bull Researcher" in qg_targets, (
            f"Quality Gate should route to Bull Researcher; got {qg_targets}"
        )

    def test_last_msg_clear_routes_to_quality_gate(self, mock_llm_client):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        # With a single analyst its Msg Clear must terminate into Quality Gate
        # — this exercises the "i == last analyst" branch in setup.py.
        g = TradingAgentsGraph(
            selected_analysts=["market"],
            config=DEFAULT_CONFIG.copy(),
        )
        edges = list(g.workflow.edges)
        last_clear_targets = [
            tgt for (src, tgt) in edges if src == "Msg Clear Market"
        ]
        assert "Quality Gate" in last_clear_targets


@pytest.mark.unit
class TestToolNodeRegistration:
    """``_create_tool_nodes`` must register tool nodes for all 7 analyst types."""

    def test_all_seven_analyst_types_have_tool_nodes(self, mock_llm_client):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        g = TradingAgentsGraph(
            selected_analysts=list(_LEGACY_ANALYSTS) + list(_A_STOCK_ANALYSTS),
            config=DEFAULT_CONFIG.copy(),
        )
        for at in _LEGACY_ANALYSTS + _A_STOCK_ANALYSTS:
            assert at in g.tool_nodes, f"no ToolNode for analyst '{at}'"

    def test_fundamentals_tool_node_includes_signal_tools(self, mock_llm_client):
        # Week 5.5 bug fix: fundamentals ToolNode must include the signal
        # tools that fundamentals_analyst.py binds to its LLM, otherwise the
        # tool call resolves at the LLM but the ToolNode can't execute it.
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        g = TradingAgentsGraph(
            selected_analysts=["fundamentals"],
            config=DEFAULT_CONFIG.copy(),
        )
        # tools_by_name is the standard LangGraph ToolNode introspection.
        fund_tools = set(g.tool_nodes["fundamentals"].tools_by_name.keys())
        # Signal tools that were added to fundamentals during the port:
        assert "get_profit_forecast" in fund_tools
        assert "get_industry_comparison" in fund_tools

    def test_hot_money_tool_node_includes_signal_tools(self, mock_llm_client):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        g = TradingAgentsGraph(
            selected_analysts=["hot_money"],
            config=DEFAULT_CONFIG.copy(),
        )
        hm_tools = set(g.tool_nodes["hot_money"].tools_by_name.keys())
        for required in (
            "get_hot_stocks", "get_northbound_flow", "get_concept_blocks",
            "get_fund_flow", "get_dragon_tiger_board", "get_industry_comparison",
        ):
            assert required in hm_tools, (
                f"hot_money ToolNode missing {required} — "
                "would break LLM tool calls at runtime"
            )

    def test_lockup_tool_node_includes_lockup_expiry(self, mock_llm_client):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        g = TradingAgentsGraph(
            selected_analysts=["lockup"],
            config=DEFAULT_CONFIG.copy(),
        )
        lockup_tools = set(g.tool_nodes["lockup"].tools_by_name.keys())
        assert "get_lockup_expiry" in lockup_tools


@pytest.mark.unit
class TestInitialStateInitializesNewFields:
    """Propagator must initialize the 4 new state fields."""

    def test_propagator_initialises_a_stock_report_fields(self, mock_llm_client):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        g = TradingAgentsGraph(
            selected_analysts=["market"],
            config=DEFAULT_CONFIG.copy(),
        )
        state = g.propagator.create_initial_state("600519.SH", "2026-05-12")
        for field in (
            "policy_report", "hot_money_report", "lockup_report",
            "data_quality_summary",
        ):
            assert field in state, f"initial state missing {field}"
            assert state[field] == "", f"{field} not initialised to empty string"
