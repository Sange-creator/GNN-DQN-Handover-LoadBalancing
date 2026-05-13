# Repository Layout

This repository keeps runnable code separate from writing, reports, and generated artifacts.

## Code And Experiments

Use these locations for implementation and reproducible experiment work:

```text
src/handover_gnn_dqn/     Reusable Python package code
scripts/                  Config-driven train, evaluate, data, and figure entrypoints
tests/                    Unit, integration, and regression tests
configs/                  Experiment and scenario configuration
tools/                    One-off utility scripts for data collection or external tooling
web_dashboard/            Optional local dashboard app
run_experiment.py         Compatibility wrapper for smoke UE-only training
run_overnight.py          Compatibility wrapper for Pokhara UE-only training
run_full_training.py      Compatibility wrapper for multi-scenario UE-only training
```

Keep new reusable handover logic in `src/handover_gnn_dqn/`. Prefer `scripts/` for command-line entrypoints and `tests/` for acceptance coverage.

## Documentation And Artifacts

Use these locations for non-code material:

```text
docs/thesis/              Thesis chapters and abstract
docs/reports/             Evaluation reports and narrative summaries
docs/guides/              Human-readable guides and DOCX exports
docs/references/          Papers and external reference PDFs
docs/prompts/             Agent prompts and drafting prompts
docs/sites/               Presentation/demo websites that are not core training code
results/logs/             Local run logs
results/runs/             Generated experiment runs, ignored by git
results/archive_prefix/   Archived pre-refactor diagnostic outputs, ignored by git
```

Do not cite old archived outputs as final results. Treat `results/archive_prefix/pre_refactor_2026-05-09/` as diagnostic evidence only.
