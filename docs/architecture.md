# TradingAgents architecture (post-Codex provider, 2026-05)

Three Mermaid diagrams at increasing zoom levels: top-level provider
dispatch, the Codex-subscription provider's internals, and the
LangGraph node flow.

The graph layer is intentionally provider-agnostic — every provider
honours the same three contracts (`bind_tools` returns OpenAI-format
`tool_calls`, `with_structured_output` returns a Pydantic instance,
`invoke()` is synchronous) so the orchestrator at the top doesn't
need to know whether a given run uses Platform API, Anthropic, or a
ChatGPT subscription.

## 1. Top-level dispatch

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#fff','primaryTextColor':'#111','primaryBorderColor':'#555','lineColor':'#555','fontFamily':'monospace'}}}%%
flowchart TB
    subgraph entry["User Entry"]
        CLI["cli/main.py<br/>(interactive)"]
        PROG["main.py / TradingAgentsGraph<br/>(programmatic)"]
    end

    subgraph orchestrator["tradingagents/graph/trading_graph.py — TradingAgentsGraph"]
        QUICK["quick_thinking_llm<br/>(analysts + reflector +<br/>signal processor)"]
        DEEP["deep_thinking_llm<br/>(researchers + managers +<br/>risk debators + trader)"]
    end

    FACTORY["llm_clients/factory.py<br/>create_llm_client(provider, ...)"]

    subgraph providers["Providers"]
        direction LR
        P1["OpenAIClient<br/>api.openai.com"]
        P2["AnthropicClient<br/>api.anthropic.com"]
        P3["GoogleClient<br/>generativelang.googleapis.com"]
        P4["AzureOpenAIClient<br/>*.azure.com"]
        P5["OpenAI-compatible family<br/>xAI / qwen / glm / minimax /<br/>deepseek / openrouter / ollama"]
        P6["★ CodexClient<br/>chatgpt.com/backend-api/codex<br/>(ChatGPT subscription)"]
    end

    CLI --> orchestrator
    PROG --> orchestrator
    QUICK -->|create_llm_client| FACTORY
    DEEP -->|create_llm_client| FACTORY
    FACTORY --> P1
    FACTORY --> P2
    FACTORY --> P3
    FACTORY --> P4
    FACTORY --> P5
    FACTORY --> P6

    classDef new fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#111
    class P6 new
```

## 2. Codex provider internals

Two files implement everything Codex-specific. Orange blocks are
LangChain method overrides that adapt the WHAM Responses-API
divergences from native OpenAI. Green blocks are the concurrency /
persistence safety guards added after the GAN review.

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#fff','primaryTextColor':'#111','primaryBorderColor':'#555','lineColor':'#555','fontFamily':'monospace'}}}%%
flowchart LR
    subgraph client_module["codex_client.py"]
        CC["CodexClient<br/>(BaseLLMClient)"]
        CC_INIT["__init__:<br/>eager auth load (fail-fast)"]
        CC_GETLLM["get_llm():<br/>pins CODEX_BASE_URL,<br/>injects ChatGPT-Account-Id,<br/>use_responses_api=True"]
        CHAT["ChatCodexSubscription<br/>(NormalizedChatOpenAI subclass)"]
        GEN["_generate() override:<br/>aggregates _stream_responses<br/>(WHAM requires stream=true)"]
        PAYLOAD["_get_request_payload() override:<br/>1) rotate root_client.api_key<br/>2) call _adapt_payload_for_wham"]
        ADAPT["_adapt_payload_for_wham():<br/>• lift system → instructions<br/>• force stream=true<br/>• force store=false"]
    end

    subgraph auth_module["codex_auth.py"]
        LOAD["load(path)<br/>→ process-wide singleton cache<br/>per resolved auth path"]
        CRED["CodexCredentials"]
        ACCESS["access_token()<br/>JWT exp check, refresh if<br/>under 5min remaining"]
        FORCE["force_refresh()<br/>(CLI preflight uses this)"]
        REFRESH["_refresh_locked()<br/>POST oauth/token<br/>+ bad-token loop guard"]
        RELOAD["_reload_if_disk_newer()<br/>pick up CLI-rotated tokens"]
        PERSIST["_persist_locked()<br/>atomic O_CREAT|O_EXCL 0o600<br/>+ os.replace"]
        TLOCK["threading.Lock<br/>(in-process)"]
        FLOCK["fcntl.flock<br/>(cross-process)"]
    end

    AUTHFILE[("~/.codex/auth.json<br/>mode 0o600<br/>tokens + account_id")]
    OAUTH[("auth.openai.com/<br/>oauth/token")]
    WHAM[("chatgpt.com/<br/>backend-api/codex/responses<br/>WHAM Responses API")]
    CODEX_CLI["codex CLI<br/>(separate process,<br/>shares auth.json)"]

    CC --> CC_INIT --> LOAD
    CC --> CC_GETLLM --> CHAT
    CHAT --> GEN
    CHAT --> PAYLOAD
    PAYLOAD --> ACCESS
    PAYLOAD --> ADAPT
    LOAD --> CRED
    CRED --> ACCESS
    CRED --> FORCE
    ACCESS --> TLOCK
    FORCE --> TLOCK
    TLOCK --> FLOCK
    FLOCK --> RELOAD
    FLOCK --> REFRESH
    REFRESH --> PERSIST
    RELOAD <-->|read| AUTHFILE
    PERSIST -->|write| AUTHFILE
    REFRESH -->|POST refresh_token| OAUTH
    CODEX_CLI <-->|reads + writes| AUTHFILE
    GEN -->|POST /responses<br/>Bearer + Account-Id| WHAM

    classDef override fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#111
    classDef safety fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#111
    class GEN,PAYLOAD,ADAPT override
    class TLOCK,FLOCK,RELOAD,PERSIST safety
```

## 3. LangGraph node flow

Provider-agnostic. Orange nodes use `with_structured_output` for typed
Pydantic results; blue nodes use plain `invoke()` (often with
`bind_tools`). The Codex provider satisfies both contracts, which is
why it slots in here without touching any graph code.

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#fff','primaryTextColor':'#111','primaryBorderColor':'#555','lineColor':'#555','fontFamily':'monospace'}}}%%
flowchart TB
    INIT([analysis start:<br/>ticker + date])

    subgraph analysts["Analysts (quick_thinking_llm + bind_tools)"]
        direction LR
        A1["market_analyst"]
        A2["social_media_analyst"]
        A3["news_analyst"]
        A4["fundamentals_analyst"]
        A5["a-stock:<br/>hot_money_tracker<br/>lockup_watcher<br/>policy_analyst"]
    end

    TOOLS[("LangGraph ToolNode<br/>yfinance / alpha_vantage /<br/>reddit / news / a-stock signals")]
    QGATE["quality_gate"]

    subgraph debate["Research debate (deep_thinking_llm)"]
        BULL["bull_researcher"]
        BEAR["bear_researcher"]
        RM["research_manager<br/>(with_structured_output)"]
    end

    subgraph risk["Risk debate (deep_thinking_llm)"]
        RAG["aggressive_debator"]
        RNT["neutral_debator"]
        RCO["conservative_debator"]
    end

    TRADER["trader<br/>(with_structured_output)"]
    PM["portfolio_manager<br/>(with_structured_output)"]
    SIG["signal_processor<br/>(quick_thinking_llm)"]
    REFL["reflector<br/>(quick_thinking_llm)"]
    MEM[("~/.tradingagents/memory/<br/>trading_memory.md")]
    OUT([buy / sell / hold])

    INIT --> analysts
    analysts <-->|tool_calls| TOOLS
    analysts --> QGATE
    QGATE --> BULL
    QGATE --> BEAR
    BULL <--> BEAR
    BULL --> RM
    BEAR --> RM
    RM --> RAG
    RM --> RNT
    RM --> RCO
    RAG <--> RNT
    RNT <--> RCO
    RAG --> TRADER
    RNT --> TRADER
    RCO --> TRADER
    TRADER --> PM
    PM --> SIG
    SIG --> OUT
    PM --> REFL
    REFL <-->|read history,<br/>write reflection| MEM

    classDef llm fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#111
    classDef structured fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#111
    class A1,A2,A3,A4,A5,BULL,BEAR,RAG,RNT,RCO,SIG,REFL llm
    class RM,TRADER,PM structured
```

## Contracts that keep providers interchangeable

| # | Contract | Native OpenAI | Codex (WHAM) |
|---|---|---|---|
| 1 | `bind_tools()` returns OpenAI-format `tool_calls` | direct | via `_stream_responses` aggregation |
| 2 | `with_structured_output()` returns Pydantic instance | direct | direct (same Responses API) |
| 3 | `invoke()` synchronous semantics | direct | `_generate` collapses stream chunks |
| 4 | Token rotation invisible to LangChain | n/a (long-lived API key) | `_get_request_payload` writes `root_client.api_key` |
| 5 | Multiple in-process LLM instances share refresh state | n/a | `load()` is a per-path singleton |
| 6 | Doesn't race with the standalone `codex` CLI | n/a | `fcntl.flock` + `_reload_if_disk_newer` |
| 7 | Credential file mode `0o600` at all times | n/a | `os.open(O_CREAT\|O_EXCL, 0o600)` |
| 8 | Auth failures surface before the graph starts | env-var check | `force_refresh()` preflight in CLI |
