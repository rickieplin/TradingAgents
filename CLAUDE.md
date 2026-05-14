# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TradingAgents is a multi-agent LLM financial-trading framework built on **LangGraph**. A `TradingAgentsGraph` orchestrates specialised agents (analysts → researchers → trader → risk debators → portfolio manager) that debate and emit a buy/sell/hold decision for a ticker on a given date.

## Commands

### Install / run
```bash
pip install .                           # install package + CLI entry point
tradingagents                           # interactive CLI (questionary prompts)
python -m cli.main                      # equivalent, from source
python main.py                          # programmatic one-shot example (NVDA)
```

### Tests
```bash
pytest                                  # full suite (pyproject sets testpaths=["tests"])
pytest tests/test_memory_log.py         # single file
pytest tests/test_memory_log.py::test_name   # single test
pytest -m unit                          # by marker: unit | integration | smoke
```
`tests/conftest.py` autouses a `_dummy_api_keys` fixture that sets `placeholder` for every provider env var, so unit tests run offline without hanging on real API calls. Use `mock_llm_client` to stub `tradingagents.llm_clients.factory.create_llm_client`.

### Docker
```bash
docker compose run --rm tradingagents
docker compose --profile ollama run --rm tradingagents-ollama
```

## Architecture

### Top-level packages
- `tradingagents/graph/` — LangGraph orchestration. `trading_graph.py::TradingAgentsGraph` wires LLM clients, tool nodes, the workflow setup (`setup.GraphSetup`), the propagator, the reflector, and the signal processor. `checkpointer.py` adds opt-in SQLite resume.
- `tradingagents/agents/` — agent implementations grouped by role: `analysts/` (market, sentiment, news, fundamentals), `researchers/` (bull/bear), `managers/` (research, portfolio), `risk_mgmt/` (aggressive/neutral/conservative debators), `trader/`. `utils/agent_utils.py` exposes the tool functions wired into LangGraph `ToolNode`s; `utils/memory.py` is the cross-run decision log; `schemas.py` defines structured-output Pydantic models.
- `tradingagents/dataflows/` — data-vendor layer. `interface.py` is the abstract surface; `y_finance.py`, `alpha_vantage*.py`, `reddit.py`, `stocktwits.py`, `yfinance_news.py` are concrete vendors. Vendor selection is config-driven (`data_vendors` category-level, `tool_vendors` per-tool override).
- `tradingagents/llm_clients/` — provider abstraction. `factory.create_llm_client(provider, model, base_url, **kwargs)` returns a client whose `.get_llm()` yields a LangChain chat model. `capabilities.py` maps provider/model → which structured-output / tool-choice features are supported; `model_catalog.py` is the curated model list; `api_key_env.py` resolves the env var name per provider; `validators.py` rejects unknown models early.
- `cli/` — `typer` + `questionary` + `rich` interactive CLI. `main.py` is the entry point (`MessageBuffer` orchestrates the live display); `utils.py` handles ticker/provider/model prompts; `stats_handler.py` is a LangChain callback that tracks LLM/tool usage for the panel.

### Configuration model
`tradingagents/default_config.py` is the single source of truth. `DEFAULT_CONFIG` is built by `_apply_env_overrides()` over a dict literal — `TRADINGAGENTS_*` env vars override the matching keys with type coercion driven by the existing default's type (bool / int / float / str). When adding a new env-overrideable config key, extend the `_ENV_OVERRIDES` map; **no entry-point script changes are needed**.

`backend_url` defaults to `None` so each provider's client uses its own default endpoint — do not set a provider-specific URL here, it leaks across providers (an OpenAI `/v1` once got forwarded to Gemini and produced malformed request URLs).

`set_config()` in `tradingagents/dataflows/config.py` performs partial updates that preserve sibling defaults — do not replace nested dicts wholesale.

### Persistence
- **Decision log** (always on): `~/.tradingagents/memory/trading_memory.md`. After a run completes, the next run for the same ticker fetches the realised return and alpha vs. the resolved benchmark, generates a reflection, and injects same-ticker history + cross-ticker lessons into the Portfolio Manager prompt. Override path with `TRADINGAGENTS_MEMORY_LOG_PATH`; cap entries with `memory_log_max_entries` (pending entries are never pruned).
- **Checkpoints** (opt-in via `--checkpoint` or `checkpoint_enabled=True`): per-ticker SQLite DBs at `~/.tradingagents/cache/checkpoints/<TICKER>.db`. Cleared on successful completion; `--clear-checkpoints` resets before a run.

### Benchmark resolution (alpha calculation)
`TradingAgentsGraph._resolve_benchmark()` picks the alpha benchmark: explicit `benchmark_ticker` config wins; otherwise suffix-matched against `benchmark_map` (`.NS`→`^NSEI`, `.T`→`^N225`, `.HK`→`^HSI`, `.L`→`^FTSE`, `.TO`→`^GSPTSE`, `.AX`→`^AXJO`, `.BO`→`^BSESN`, default `SPY`). The empty-suffix fallback is intentional for US tickers with dots like `BRK.B` — alpha is computed in USD so SPY is correct.

### Provider quirks worth remembering
- DeepSeek V4/reasoner and MiniMax M2.x reject `tool_choice`; the binding flow consults `llm_clients/capabilities.py` and skips it automatically. Don't unconditionally pass `tool_choice` when adding new structured-output flows.
- Qwen, GLM, and MiniMax have dual-region keys (e.g. `DASHSCOPE_API_KEY` international vs. `DASHSCOPE_CN_API_KEY` China). The CLI prompts for region; the API-key env var name comes from `api_key_env.py`.
- Ollama: default `http://localhost:11434/v1`; `OLLAMA_BASE_URL` points at a remote server. CLI offers a "Custom model ID" option for arbitrary `ollama pull`-ed models.

### Tickers and path safety
Ticker strings reach the filesystem (cache dir, checkpoint DB path, memory log). Always route ticker components through `tradingagents.dataflows.utils.safe_ticker_component` before joining onto a path — exchange suffixes (`.SH`, `.HK`, `.T`, etc.) are legitimate, but `..`/`/` are not.

## Agent skills

### Issue tracker

GitHub Issues on `rickieplin/TradingAgents` via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`); missing labels are created on first use. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — `CONTEXT.md` and `docs/adr/` at the repo root (not yet authored). See `docs/agents/domain.md`.
