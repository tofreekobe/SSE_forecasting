# -*- coding: utf-8 -*-
"""Audit whether the SSE repository is ready to publish to GitHub."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRET_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
REQUIRED_FILES = [
    "README.md",
    "requirements-pai.txt",
    "requirements-diagnostics.txt",
    "src/dataset/forecast_contract.py",
    "src/dataset/package_forecast_contract.py",
    "src/models/small_forecast_net.py",
    "scripts/train_forecast_model.py",
    "scripts/run_dsw_experiment_matrix.sh",
    "scripts/serve_demo_gui.py",
    "docs/final_paper_manuscript_zh.md",
    "docs/final_paper_manuscript_zh.docx",
    "docs/final_conference_paper_outline_zh.md",
    "docs/paper_result_tables_current.md",
    "docs/complete_usage_guide_zh.md",
    "docs/model_demo_usage.md",
    "docs/github_publish_guide_zh.md",
    "docs/final_completion_audit_zh.md",
    "docs/full_dataset_package_audit.md",
]
FORBIDDEN_TRACKED_PREFIXES = [
    "data/",
    "hf_dataset_package/",
    "paper/",
    "forecast_training",
    "small_overfit",
    "checkpoints/",
    "logs/",
    "demo_pages/",
    "dsw_results/",
    "diagnostics_full_local/",
]
EVIDENCE_CHECKS = {
    "README.md": ["GO_WITH_CHANGES", "50-step future slip forecasting"],
    "docs/final_paper_manuscript_zh.md": ["6000", "74.202", "segmented residual", "GNSS-only"],
    "docs/paper_result_tables_current.md": ["segmented_residual", "ablate_gnss_only", "PASS", "FAIL"],
    "docs/model_demo_usage.md": ["serve_demo_gui.py", "inversion proxy", "not a paper-grade inversion model"],
    "docs/complete_usage_guide_zh.md": ["serve_demo_gui.py", "GitHub"],
    "docs/github_publish_guide_zh.md": ["check_release_ready.py", "git push"],
    "docs/final_completion_audit_zh.md": ["6000", "74.202", "14", "GitHub remote"],
}


def run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def tracked_files() -> list[str]:
    proc = run_git(["ls-files"])
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def git_status() -> list[str]:
    proc = run_git(["status", "--short"])
    return [line for line in proc.stdout.splitlines() if line.strip()]


def git_remotes() -> list[str]:
    proc = run_git(["remote", "-v"])
    return [line for line in proc.stdout.splitlines() if line.strip()]


def scan_secrets(files: list[str]) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    for rel in files:
        path = PROJECT_ROOT / rel
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            for match in SECRET_RE.findall(line):
                hits.append({"file": rel, "line": line_no, "token_prefix": match[:10] + "..."})
    return hits


def check_ignored_paths(paths: list[str]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for rel in paths:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        proc = run_git(["check-ignore", "-v", rel], check=False)
        results.append(
            {
                "path": rel,
                "exists": True,
                "ignored": proc.returncode == 0,
                "rule": proc.stdout.strip(),
            }
        )
    return results


def check_evidence_strings() -> list[dict[str, object]]:
    missing: list[dict[str, object]] = []
    for rel, needles in EVIDENCE_CHECKS.items():
        path = PROJECT_ROOT / rel
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        for needle in needles:
            if needle not in text:
                missing.append({"file": rel, "missing": needle})
    return missing


def build_report(require_clean: bool) -> dict[str, object]:
    files = tracked_files()
    status = git_status()
    missing_required = [rel for rel in REQUIRED_FILES if not (PROJECT_ROOT / rel).exists()]
    forbidden_tracked = [
        rel
        for rel in files
        if any(rel == prefix.rstrip("/") or rel.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES)
    ]
    ignored_local_private = check_ignored_paths(["data", "hf_dataset_package", "paper", "demo_pages", "dsw_results"])
    not_ignored_private = [item for item in ignored_local_private if not item["ignored"]]
    secret_hits = scan_secrets(files)
    missing_evidence = check_evidence_strings()
    remotes = git_remotes()

    failures = []
    if require_clean and status:
        failures.append({"kind": "dirty_worktree", "items": status})
    if missing_required:
        failures.append({"kind": "missing_required_files", "items": missing_required})
    if forbidden_tracked:
        failures.append({"kind": "forbidden_tracked_paths", "items": forbidden_tracked})
    if not_ignored_private:
        failures.append({"kind": "private_paths_not_ignored", "items": not_ignored_private})
    if secret_hits:
        failures.append({"kind": "secret_token_hits", "items": secret_hits})
    if missing_evidence:
        failures.append({"kind": "missing_evidence_strings", "items": missing_evidence})

    warnings = []
    if not remotes:
        warnings.append("No git remote is configured; add a GitHub repository before pushing.")

    return {
        "ok": not failures,
        "tracked_file_count": len(files),
        "git_status_count": len(status),
        "remote_count": len(remotes),
        "remotes": remotes,
        "ignored_local_private_paths": ignored_local_private,
        "warnings": warnings,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true", help="Do not fail on uncommitted local changes.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    report = build_report(require_clean=not args.allow_dirty)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Release ready: {report['ok']}")
        print(f"Tracked files: {report['tracked_file_count']}")
        print(f"Git status entries: {report['git_status_count']}")
        print(f"Remote count: {report['remote_count']}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for failure in report["failures"]:
            print(f"FAIL: {failure['kind']}")
            for item in failure["items"][:20]:
                print(f"  - {item}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
