# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues on **`rickieplin/TradingAgents`** (the user's fork). Use the `gh` CLI for all operations.

Note: `upstream` (`TauricResearch/TradingAgents`) is the canonical project, but day-to-day work is tracked on the fork. Don't open issues on `upstream` from these skills unless explicitly asked — pass `--repo TauricResearch/TradingAgents` only when the user specifies it.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

`gh` infers the repo from `git remote -v` automatically when run inside the clone. Since this repo has both `origin` (fork) and `upstream` (canonical), `gh` defaults to `origin` — which is what we want. If `gh` ever prompts to pick, choose `rickieplin/TradingAgents`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue on `rickieplin/TradingAgents`.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.
