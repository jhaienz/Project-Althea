# Supervised auto mode for AI coding agent orchestration

When Althea delegates work to external AI coding agents (Claude Code, Codex, Aider, etc.), she operates in "supervised auto" mode: the coding agent works on a separate git branch, and Althea reports what changed and asks the user before merging.

This is a deliberate trade-off between full automation (risky — coding agents can make destructive changes) and manual approval gates (slow — defeats the purpose of voice-driven automation). Supervised auto preserves the "hands-free" experience while keeping destructive actions reversible. The git branch acts as a natural sandbox.

Considered: full auto (no review) — rejected because coding agents can make unpredictable changes to a codebase. Approval gates before execution — rejected because requiring verbal approval for every file change defeats the voice assistant workflow.
