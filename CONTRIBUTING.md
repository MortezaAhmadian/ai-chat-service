# Git & GitHub workflow

## Branching: trunk-based with short-lived branches

`main` is always deployable and protected (PR required, CI must pass, 1 approval, linear history).

```bash
git switch main && git pull --rebase
git switch -c feat/batch-endpoint          # feat/, fix/, chore/, docs/, refactor/, test/
# ...small commits...
git fetch origin && git rebase origin/main # keep your branch current, resolve conflicts locally
git push -u origin feat/batch-endpoint     # open a PR
# after review fixes:
git push --force-with-lease                # never plain --force on a shared branch
```

Merge with **squash** (one clean commit per PR on `main`) unless the individual commits are
meaningful on their own, in which case rebase-merge.

## Commits: Conventional Commits

```text
feat(chat): add SSE streaming endpoint
fix(cache): release single-flight slot when leader is cancelled
refactor(llm): extract provider behind LLMClient protocol
test(resilience): cover half-open probe cancellation
chore(ci): cache pip dependencies
feat(api)!: rename `reply.kind` to `reply.action.type`   # ! = breaking change
```

Why: readable history, automatic changelogs and semantic version bumps (feat → minor,
fix → patch, `!` → major).

## Useful commands you should be fluent in

| Goal | Command |
|---|---|
| Tidy commits before a PR | `git rebase -i origin/main` (squash/fixup/reword) |
| Fix the previous commit | `git commit --amend` (only if not pushed, or push with `--force-with-lease`) |
| Auto-squash a fix into an older commit | `git commit --fixup <sha>` then `git rebase -i --autosquash origin/main` |
| Find which commit broke a test | `git bisect start; git bisect bad; git bisect good <sha>; git bisect run pytest -q tests/x.py` |
| Undo a pushed commit safely | `git revert <sha>` (new commit; never rewrite shared history) |
| Recover "lost" work | `git reflog` then `git switch -c rescue <sha>` |
| Park unfinished work | `git stash push -m "wip"` / `git stash pop` |
| Port one fix to a release branch | `git cherry-pick <sha>` |
| Who changed this line and why | `git blame -w -C <file>` then `git show <sha>` |

## Secrets

Never commit `.env` or keys. `pre-commit` runs `detect-private-key`. If a key is pushed,
**rotate it first**; rewriting history (`git filter-repo`) is secondary because forks and
clones already have it.

## PR checklist

See `.github/pull_request_template.md`. CI runs ruff, mypy --strict, pytest on 3.11 and 3.12,
and a Docker build; the branch cannot merge unless all pass.
