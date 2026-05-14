# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Canonical role     | Label in our tracker | Meaning                                  |
| ------------------ | -------------------- | ---------------------------------------- |
| `needs-triage`     | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`       | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`  | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`  | `ready-for-human`    | Requires human implementation            |
| `wontfix`          | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

## Lazy label creation

As of setup, only `wontfix` exists on `rickieplin/TradingAgents` (it ships with the GitHub default label set). The other four labels do **not** exist yet. When a skill needs to apply one for the first time, create it on demand with `gh label create`:

```bash
gh label create needs-triage   --color "FBCA04" --description "Maintainer needs to evaluate"        2>/dev/null || true
gh label create needs-info     --color "D4C5F9" --description "Waiting on reporter for more information" 2>/dev/null || true
gh label create ready-for-agent --color "0E8A16" --description "Fully specified, AFK-ready"        2>/dev/null || true
gh label create ready-for-human --color "1D76DB" --description "Needs human implementation"        2>/dev/null || true
```

The `|| true` swallows the "already exists" error so re-running is safe. Edit the right-hand column above if you ever change the vocabulary.
