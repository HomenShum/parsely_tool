# README documentation history

## 2026-10-16 — Clarify the documentation-only project state

Replace runnable-looking setup instructions with the source-backed state: the reviewed revision tracks only a README, while the modules, manifests, examples and tests are absent. Retain the intended feature scope as planned capabilities and provide no unsupported install or runtime route. Clean the entrypoint's malformed fences and NUL-interleaved tail.

**Scope:** [README.md](../../README.md). No application, dependency, configuration, data, provider or deployment behavior changes.
**Provenance:** The [original README](https://github.com/HomenShum/parsely_tool/blob/a125ce5d44e7257442b18768af9bd63a4822fbbc/README.md) is preserved by its immutable Git revision and the review's exact original-byte evidence; no history is backfilled.
**Verification:** The complete pinned tree and original blob establish the documentation gap. Proposed document links, Markdown fences and absence of NUL bytes are checked offline. Installation, runtime, providers, storage writes and test or model results are not certified.
