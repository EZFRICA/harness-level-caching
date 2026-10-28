---
name: house-rules
description: The owner's standing rules for CAMA (no commit without the owner, English in the repository and French with the owner, no em or en dash, Markdown files of 100 lines at most, secrets never shown, uv for Python, port 8797, paid calls capped, sibling projects read only). Use before any command, edit, document, commit, paid model call, or message to the owner.
when_to_use: Starting a task; touching .env; writing any text that lands in the repository; about to commit, restart the service, or call a paid model; reporting to the owner.
---

# House rules

The owner set these; a request that seems to break one is a question to ask, not a choice to
make.

| rule | how it is held |
|---|---|
| Nothing is committed, pushed or branched without the owner saying so | a finished task is reported as uncommitted |
| The repository is in English (code, docs, interface); the owner is answered in French, with tu | review |
| No em dash and no en dash, anywhere in the repository | `tests/test_trace.py`, the ledger refuses them |
| A Markdown file has 100 lines at most; split it | `tests/test_trace.py`; generated pages split themselves |
| A key, token or `.env` value is never printed, logged or stored | the ledger refuses secret-shaped text; check config by variable name only |
| Python on the host runs through `uv` (`uv run ...`, `uv add ...`) | the lockfile |
| CAMA owns port 8797 and its compose services; nothing else is started, stopped or probed | compose file |
| Paid model calls are capped by `HC_MAX_USD` (2 USD per run); scale and stress use the echo model | the budget guard |
| `../` projects are read for ideas, never changed, never copied wholesale | review |

## A change is done when

1. `uv run pytest -q` is green and any new guarantee has a mutant that the suite catches
   (`review-a-change`);
2. it ran live in the container, with its run ids, or it is reported as not run live;
3. the ledger has its event (`keep-the-trace`).

## Every report to the owner

French, tu, what ran with its numbers and run ids, what did not run, whether anything is
committed (by default nothing), no secret or network address.
