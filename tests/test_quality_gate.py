"""Tests for the Quality Gate node's hard-check logic.

The Quality Gate sits between the last analyst and the bull/bear debate
and grades each analyst report from A (complete) to F (empty). The
grading is deterministic — driven by length, presence of failure
markers, presence of a markdown table, and ``[数据缺失`` count — so it
can be tested without an LLM.

The LLM-review path (Layer 2) is exercised at a contract level: when
``fail_count >= 4`` the LLM is skipped, and the node always writes a
``data_quality_summary`` field to state.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.quality_gate import (
    FAILURE_MARKERS,
    MIN_REPORT_LENGTH,
    REPORT_FIELDS,
    _hard_check_report,
    create_quality_gate,
)


# ---------------------------------------------------------------------------
# _hard_check_report — five grade bands
# ---------------------------------------------------------------------------


def _good_report() -> str:
    """A long report with a markdown table and no failure markers."""
    body = "技术分析报告。" + ("分析内容很详细。" * 50)
    table = "\n\n| 指标 | 值 |\n| --- | --- |\n| RSI | 65 |\n"
    return body + table


@pytest.mark.unit
class TestHardCheckGrades:
    """Each of the five grade bands has a clear trigger."""

    def test_empty_report_is_grade_f(self):
        grade, _detail = _hard_check_report("market", "")
        assert grade == "F"

    def test_whitespace_only_report_is_grade_f(self):
        grade, _detail = _hard_check_report("market", "   \n\n  ")
        assert grade == "F"

    def test_short_report_is_grade_d(self):
        short = "太短了。" * 5  # well under MIN_REPORT_LENGTH (200 chars)
        assert len(short) < MIN_REPORT_LENGTH
        grade, _detail = _hard_check_report("market", short)
        assert grade == "D"

    def test_mostly_failure_markers_is_grade_d(self):
        # If failure markers dominate the report content, grade D.
        marker = FAILURE_MARKERS[0]  # "无法获取"
        # Build a report where stripping the markers leaves <MIN_REPORT_LENGTH.
        report = (marker + " " * 5) * 50  # only markers + spaces
        grade, _detail = _hard_check_report("market", report)
        assert grade == "D"

    def test_many_missing_data_markers_is_grade_c(self):
        report = _good_report() + "\n[数据缺失: A]\n[数据缺失: B]\n[数据缺失: C]\n"
        grade, _detail = _hard_check_report("market", report)
        assert grade == "C"

    def test_missing_table_is_grade_b(self):
        # Long report, no failure markers, no markdown table.
        report = "技术分析报告，" + ("非常详尽的分析内容，" * 50)
        assert "|" not in report and "---" not in report
        grade, _detail = _hard_check_report("market", report)
        assert grade == "B"

    def test_one_missing_data_marker_is_grade_b(self):
        report = _good_report() + "\n[数据缺失: 某项]\n"
        grade, _detail = _hard_check_report("market", report)
        assert grade == "B"

    def test_complete_report_is_grade_a(self):
        grade, _detail = _hard_check_report("market", _good_report())
        assert grade == "A"


# ---------------------------------------------------------------------------
# create_quality_gate — node-level behaviour
# ---------------------------------------------------------------------------


def _state_with_reports(**reports) -> dict:
    """Build an agent state with the seven report fields set."""
    state = {
        "trade_date": "2026-05-12",
        "company_of_interest": "600519.SH",
    }
    for field in REPORT_FIELDS.values():
        state[field] = reports.get(field, "")
    return state


@pytest.mark.unit
class TestQualityGateNode:
    """The node always writes data_quality_summary; LLM skip logic works."""

    def test_node_writes_data_quality_summary(self):
        # LLM is mocked; we just check the field appears in the returned delta.
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="LLM review text")
        node = create_quality_gate(llm)

        state = _state_with_reports(market_report=_good_report())
        result = node(state)
        assert "data_quality_summary" in result
        assert isinstance(result["data_quality_summary"], str)
        assert result["data_quality_summary"].strip() != ""

    def test_summary_includes_hard_check_lines(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="ok")
        node = create_quality_gate(llm)

        # Mix of grades so the hard-check section has variety.
        state = _state_with_reports(
            market_report=_good_report(),     # A
            sentiment_report="",              # F
            news_report="太短了" * 5,         # D
        )
        result = node(state)
        summary = result["data_quality_summary"]
        # The Chinese analyst names appear in the hard-summary block.
        assert "技术分析师" in summary
        assert "[A]" in summary
        assert "[F]" in summary

    def test_llm_review_skipped_when_4_or_more_failures(self):
        # Per the fork's design decision #8: when 4+ reports fail hard
        # checks, skip the LLM call to save tokens.
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="should not be called")
        node = create_quality_gate(llm)

        # Make 4 of the 7 reports empty (F grade).
        state = _state_with_reports(
            market_report="", sentiment_report="", news_report="",
            fundamentals_report="",
            # The remaining 3 are good
            policy_report=_good_report(),
            hot_money_report=_good_report(),
            lockup_report=_good_report(),
        )
        node(state)
        assert llm.invoke.call_count == 0, (
            "LLM review must be skipped when >=4 reports fail hard checks"
        )

    def test_llm_review_runs_when_few_failures(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="LLM review")
        node = create_quality_gate(llm)

        # All 7 reports good — LLM should be called once.
        state = _state_with_reports(
            **{field: _good_report() for field in REPORT_FIELDS.values()}
        )
        node(state)
        assert llm.invoke.call_count == 1

    def test_node_resilient_to_llm_failure(self):
        # If the LLM raises, the node must still return a summary (with the
        # exception captured) instead of crashing the whole graph.
        # Note: enough good reports needed so we actually reach the LLM
        # call (the >=4 failure path skips the LLM and never raises).
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("api down")
        node = create_quality_gate(llm)

        state = _state_with_reports(
            **{field: _good_report() for field in REPORT_FIELDS.values()}
        )
        result = node(state)
        assert "data_quality_summary" in result
        assert "LLM 复审失败" in result["data_quality_summary"]

    def test_node_handles_missing_report_fields_gracefully(self):
        # The fork's design decision #2: downstream readers use state.get()
        # so partial states don't KeyError. The node should mirror this.
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="ok")
        node = create_quality_gate(llm)

        # State missing several report fields entirely.
        state = {
            "trade_date": "2026-05-12",
            "company_of_interest": "600519.SH",
            "market_report": _good_report(),
            # Other fields deliberately omitted
        }
        # Must not raise KeyError.
        result = node(state)
        assert "data_quality_summary" in result
