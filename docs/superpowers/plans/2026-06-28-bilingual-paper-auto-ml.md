# Bilingual Paper Draft and AutoML Workbench Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce Chinese and English SSE paper drafts with figures, tables, and reviewer-facing gap analysis, then package the project so the AutoResearch workbench can continue literature review and polishing when a live OpenAI provider is configured.

**Architecture:** Keep the SSE repository as the authoritative source for experimental evidence and paper artifacts. Generate manuscript files deterministically from the current evidence files, and export a mirrored research workspace into `mm/auto_ml/projects/sse_slow_slip_forecasting` without moving private data or large raw files.

**Tech Stack:** Python, python-docx, Markdown, YAML/JSON, existing SSE scripts, existing auto_ml project directory contract.

---

### Task 1: Build Bilingual Manuscript Assets

**Files:**
- Modify: `scripts/build_final_paper_docx.py`
- Create: `scripts/build_bilingual_paper_drafts.py`
- Create: `docs/submission_draft_zh.md`
- Create: `docs/submission_draft_en.md`
- Create: `docs/submission_draft_zh.docx`
- Create: `docs/submission_draft_en.docx`

- [ ] **Step 1: Add Markdown image support to the DOCX builder**

Patch `scripts/build_final_paper_docx.py` so lines matching `![caption](path)` insert a centered image at 6.05 inches wide and add an italic caption.

- [ ] **Step 2: Generate the Chinese draft from the current Chinese manuscript**

Run:

```powershell
.\.venv-cu128\Scripts\python.exe scripts\build_bilingual_paper_drafts.py --auto-ml-root "C:\Users\Administrator\Desktop\王一帆项目文件\mm\auto_ml"
```

Expected: `docs/submission_draft_zh.md` exists and contains the full current manuscript plus embedded figure references from `paper_figures_full`.

- [ ] **Step 3: Generate the English draft**

Run the same command as Step 2.

Expected: `docs/submission_draft_en.md` exists and contains English title, abstract, methods, experiments, results, discussion, limitations, references, figure captions, and all key result tables.

- [ ] **Step 4: Build DOCX deliverables**

The generator calls `build_final_paper_docx.build_docx` for both manuscripts.

Expected: `docs/submission_draft_zh.docx` and `docs/submission_draft_en.docx` exist and can be opened structurally by `python-docx`.

### Task 2: Export AutoML Research Workspace

**Files:**
- Create: `docs/auto_ml_workbench_sse_integration_zh.md`
- Create or update: `C:\Users\Administrator\Desktop\王一帆项目文件\mm\auto_ml\projects\sse_slow_slip_forecasting\**`

- [ ] **Step 1: Write an AutoML project manifest**

Create `project.yaml` with project id `sse_slow_slip_forecasting`, title `SSE Slow Slip Forecasting`, and workflow stages `literature_review`, `paper_writing`, `reviewer_gap_audit`, and `publication_export`.

- [ ] **Step 2: Mirror verified SSE evidence**

Copy or synthesize Markdown evidence into `knowledge/`, `analysis/`, and `writing/`:

```text
knowledge/research_context.md
knowledge/literature_review.md
knowledge/knowledge_gaps.md
knowledge/citation_map.json
analysis/evidence_table.md
analysis/limitation_analysis.md
writing/paper_outline.md
writing/manuscript.md
writing/manuscript_en.md
writing/references.bib
writing/tables/paper_result_tables_current.md
```

Expected: auto_ml can list and edit these files without needing access to the 74.2 GiB raw data directory.

- [ ] **Step 3: Record live OpenAI limitation**

Write `docs/auto_ml_workbench_sse_integration_zh.md` noting that `OPENAI_API_KEY` is not present in the current shell, so live OpenAI review/polish is prepared but not yet executed.

### Task 3: Add Reviewer Gap Audit

**Files:**
- Create: `docs/top_journal_reviewer_gap_audit_zh.md`

- [ ] **Step 1: List major review risks**

Include synthetic-only data, dependence on history slip, lack of uncertainty quantification, limited external baselines, possible leakage/persistence interpretation, and GNSS-only failure.

- [ ] **Step 2: Convert risks into required follow-up work**

For each risk, define a concrete experiment, expected evidence, and whether it is required before a top-journal submission or can be appendix/future work.

### Task 4: Verify

**Files:**
- Test: `scripts/check_release_ready.py`
- Test: `tests/test_hf_diagnostics.py`
- Test: `tests/test_forecast_contract.py`

- [ ] **Step 1: Run structural document checks**

Run:

```powershell
.\.venv-cu128\Scripts\python.exe scripts\build_bilingual_paper_drafts.py --check-only --auto-ml-root "C:\Users\Administrator\Desktop\王一帆项目文件\mm\auto_ml"
```

Expected: JSON reports both DOCX files exist, both Markdown files exist, and the auto_ml project package exists.

- [ ] **Step 2: Run core tests**

Run:

```powershell
.\.venv-cu128\Scripts\python.exe -m pytest tests\test_hf_diagnostics.py tests\test_forecast_contract.py --basetemp .pytest_tmp_bilingual_goal
```

Expected: 10 tests pass.

- [ ] **Step 3: Run release readiness check**

Run:

```powershell
.\.venv-cu128\Scripts\python.exe scripts\check_release_ready.py --json
```

Expected: `ok` is true or any new warnings are explained as intentional draft artifacts.
