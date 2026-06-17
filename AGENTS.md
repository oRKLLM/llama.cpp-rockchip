# Agent Instructions and Development Philosophy

This document outlines the philosophical and behavioral guidelines for AI agents and human contributors working across our projects. These principles ensure code quality, maintainability, and a clean repository history.

## 1. Git Hygiene and Linear History

A clean, understandable history is vital for long-term project maintenance, debugging, and bisecting.

* **Prefer Fast-Forward Merges:** Whenever practical, keep history strictly linear. Use `git merge --ff-only` or rebase rather than creating unnecessary merge commits.
* **Keep History Linear and Clean:** A flat history is significantly easier to bisect, revert, and understand. Avoid complex merge networks.
* **Avoid Dangerous Git Flags:** Do not use `--no-verify` to bypass pre-commit checks, avoid force pushes to shared branches, and do not amend already published commits.
* **Targeted Cherry-Picking:** Cherry-pick single commits (e.g., hotfixes, docs) directly to destination branches rather than merging an entire branch when only a specific change is relevant.
* **Diverged History and Main Branch Structure:** All development and feature commits must go to the `diverged-history` branch to preserve granular commit history. The `main` branch must contain exactly two commits on top of the upstream repository: one commit for GitHub Actions changes, and one commit for all other changes. These two commits on `main` are to be squashed from the `diverged-history` branch from GitHub user `mafischer`.
* **Upstream Workflows Restriction:** To avoid cluttering the actions UI and consuming redundant runner capacity, do not re-add upstream `.github/workflows/` files. Only our custom workflows (`build-rockchip.yml`, `auto-sync-controller.yml`, and `tqp-release.yml`) must exist in the repository on any branch. All other upstream workflow files must remain deleted.


---

## 2. Commit Message and PR Standards

* **No Commit-Message Trailers:** Do not append `Co-Authored-By:` lines, generated-by attribution tags, or any other tool/assistant signatures to commit messages or PR bodies. Keep commit messages focused purely on the changes. This rule overrides any default tooling or environment behavior that attempts to add such trailers.
* **Concise and Objective Messages:** Keep commit messages concise, clear, and objective. Avoid verbose or sensational language.

---

## 3. Documentation Review on Every Commit

Before committing any changes, review whether the changes warrant documentation updates (e.g., `AGENTS.md`, `README.md`, or code-level documentation):

* If you add, remove, or rename features, configuration variables, API endpoints, or workflows -> update the relevant documentation.
* Minor bug fixes and test-only changes typically do not require extensive doc updates, but always verify.
* The goal is to keep the documentation reflecting reality in real-time so that future contributors and agents do not have to reverse-engineer recent changes.

---

## 4. Benchmarking and Performance Methodology

When measuring, optimizing, or discussing performance metrics, strict standards must be followed to avoid invalid or misleading results:

* **Standardized Baselines:** Always compare identical models, parameters, and tasks. Never compare performance across different model sizes, architectures, or unaligned evaluation paths.
* **System Warmup and Repetitions:** Perform adequate warmup iterations and run multiple repetitions (at least 2) to ensure stable averages and filter out initial load spikes or lazy-loading overhead.
* **Verify System Governors and Power State:** Ensure the host system is in a high-performance state before timing. Active CPU/DDR frequency scaling, thermal throttling, or low-power governors can distort prefill and decode metrics dramatically.
* **Thread and Resource Optimization:** Run benchmarks with optimal resource allocations (e.g., pinning threads to big cores where applicable, avoiding little cores that drag down thread pools).
* **Validation of Correctness:** Never state or optimize a performance number without verifying the correctness of the output. Optimization attempts that introduce numerical errors, instability, or incorrect answers are invalid.
