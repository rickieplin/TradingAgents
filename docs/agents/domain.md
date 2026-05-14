# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Layout: single-context

This repo is a single domain — one Python package (`tradingagents/`) plus a CLI (`cli/`). Skills should look at the repo root:

```
/
├── CONTEXT.md            ← glossary + domain language (not yet authored)
├── docs/adr/             ← architectural decisions (not yet authored)
│   └── 0001-...md
├── tradingagents/
└── cli/
```

Neither `CONTEXT.md` nor `docs/adr/` exists yet. That's expected — they get authored lazily by `/grill-with-docs` when terminology or decisions need to crystallise.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root
- **`docs/adr/`** — read ADRs that touch the area you're about to work in

If either doesn't exist, **proceed silently**. Don't flag the absence; don't suggest creating them upfront. The producer skill (`/grill-with-docs`) creates them lazily when terms or decisions actually get resolved.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
