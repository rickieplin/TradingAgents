"""Tests for the three A-stock analyst factories.

The fork's design decision #2 says new analyst report fields use
``state.get("field", "")`` so partial states don't KeyError. These tests
also confirm each analyst:

  - returns a callable graph node
  - binds the expected set of tools to the LLM
  - writes its report into the correct state field
  - leaves the report empty while still calling tools (the "loop" branch)

We mock the LLM at ``bind_tools`` so no real network call ever happens.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from tradingagents.agents.analysts.hot_money_tracker import create_hot_money_tracker
from tradingagents.agents.analysts.lockup_watcher import create_lockup_watcher
from tradingagents.agents.analysts.policy_analyst import create_policy_analyst


def _mock_llm(*, tool_calls: list | None = None, content: str = "report body") -> MagicMock:
    """Build an LLM mock whose ``bind_tools()`` returns a *real* Runnable.

    The agents do ``chain = prompt | llm.bind_tools(tools)`` then
    ``chain.invoke(state["messages"])``.  LangChain's ``|`` builds a
    ``RunnableSequence`` that calls each step's ``.invoke()`` in turn — so
    the right-hand side of the pipe must itself be a real Runnable, not a
    bare MagicMock (otherwise the pipe falls back to a coercion that
    returns a MagicMock attribute when invoked, and ``result.content``
    becomes a MagicMock instead of our string).

    We make ``bind_tools()`` return a ``RunnableLambda`` that always emits
    a real ``AIMessage`` with the requested ``tool_calls`` + ``content``.
    """
    message = AIMessage(content=content, tool_calls=tool_calls or [])

    llm = MagicMock()
    llm.bind_tools.return_value = RunnableLambda(lambda _prompt_value: message)
    return llm


def _base_state() -> dict:
    return {
        "trade_date": "2026-05-12",
        "company_of_interest": "600519.SH",
        "messages": [("human", "600519.SH")],
    }


# ---------------------------------------------------------------------------
# Policy Analyst
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolicyAnalystFactory:
    def test_factory_returns_callable_node(self):
        llm = _mock_llm()
        node = create_policy_analyst(llm)
        assert callable(node)

    def test_policy_analyst_binds_news_tools(self):
        # Policy analyst should bind get_news + get_global_news, no more.
        llm = _mock_llm()
        node = create_policy_analyst(llm)
        node(_base_state())
        # bind_tools is called once with the tool list as positional arg.
        assert llm.bind_tools.call_count == 1
        bound_tools = llm.bind_tools.call_args[0][0]
        names = {t.name for t in bound_tools}
        assert names == {"get_news", "get_global_news"}

    def test_writes_policy_report_when_no_tool_calls(self):
        llm = _mock_llm(tool_calls=[], content="政策面利好，行业受到扶持...")
        node = create_policy_analyst(llm)
        result = node(_base_state())
        assert result["policy_report"] == "政策面利好，行业受到扶持..."

    def test_leaves_policy_report_empty_while_calling_tools(self):
        # While the analyst is still calling tools, the report must stay
        # empty — only populated once tool_calls is empty.
        llm = _mock_llm(
            tool_calls=[{"name": "get_news", "args": {}, "id": "1"}],
            content="",
        )
        node = create_policy_analyst(llm)
        result = node(_base_state())
        assert result["policy_report"] == ""


# ---------------------------------------------------------------------------
# Hot Money Tracker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHotMoneyTrackerFactory:
    def test_factory_returns_callable_node(self):
        llm = _mock_llm()
        node = create_hot_money_tracker(llm)
        assert callable(node)

    def test_hot_money_binds_all_signal_tools(self):
        # Per the Week 5.5 bug fix in the fork, hot_money must bind every
        # signal tool its prompt advertises. Missing any of these breaks
        # the LLM's tool-call resolution at runtime.
        llm = _mock_llm()
        node = create_hot_money_tracker(llm)
        node(_base_state())
        bound = {t.name for t in llm.bind_tools.call_args[0][0]}
        for required in (
            "get_stock_data", "get_news", "get_insider_transactions",
            "get_hot_stocks", "get_northbound_flow", "get_concept_blocks",
            "get_fund_flow", "get_dragon_tiger_board", "get_industry_comparison",
        ):
            assert required in bound, (
                f"hot_money_tracker missing tool binding: {required}"
            )

    def test_writes_hot_money_report_field(self):
        llm = _mock_llm(content="主力资金净流入20亿...")
        node = create_hot_money_tracker(llm)
        result = node(_base_state())
        assert result["hot_money_report"] == "主力资金净流入20亿..."

    def test_does_not_write_other_analysts_report_fields(self):
        llm = _mock_llm(content="report")
        node = create_hot_money_tracker(llm)
        result = node(_base_state())
        # The node returns a state delta — it must touch ONLY messages +
        # its own report field, not other analysts' fields.
        assert set(result.keys()) == {"messages", "hot_money_report"}


# ---------------------------------------------------------------------------
# Lockup Watcher
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLockupWatcherFactory:
    def test_factory_returns_callable_node(self):
        llm = _mock_llm()
        node = create_lockup_watcher(llm)
        assert callable(node)

    def test_lockup_binds_lockup_expiry_tool(self):
        llm = _mock_llm()
        node = create_lockup_watcher(llm)
        node(_base_state())
        bound = {t.name for t in llm.bind_tools.call_args[0][0]}
        for required in (
            "get_insider_transactions", "get_news", "get_fundamentals",
            "get_lockup_expiry",
        ):
            assert required in bound, (
                f"lockup_watcher missing tool binding: {required}"
            )

    def test_writes_lockup_report_field(self):
        llm = _mock_llm(content="未来 3 个月无重大解禁...")
        node = create_lockup_watcher(llm)
        result = node(_base_state())
        assert result["lockup_report"] == "未来 3 个月无重大解禁..."


# ---------------------------------------------------------------------------
# Cross-analyst invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrossAnalystInvariants:
    """All three new analysts follow the same return-shape contract."""

    @pytest.mark.parametrize(
        "factory,report_field",
        [
            (create_policy_analyst, "policy_report"),
            (create_hot_money_tracker, "hot_money_report"),
            (create_lockup_watcher, "lockup_report"),
        ],
    )
    def test_return_shape_always_has_messages(self, factory, report_field):
        llm = _mock_llm(content="x")
        node = factory(llm)
        result = node(_base_state())
        assert "messages" in result
        assert len(result["messages"]) == 1  # the AIMessage from the chain
        assert report_field in result
