# -*- coding: utf-8 -*-
"""Run DeepSeek review/polish passes through the auto_ml ModelGateway."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTO_PROJECT_ID = "sse_slow_slip_forecasting"


def read_text(path: Path, limit: int | None = None) -> str:
    text = path.read_text(encoding="utf-8")
    return text[:limit] if limit else text


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_gateway(auto_ml_root: Path):
    sys.path.insert(0, str(auto_ml_root / "src"))
    from seismo_research_workbench.config import load_settings
    from seismo_research_workbench.core.db import Database
    from seismo_research_workbench.model_gateway import ModelGateway

    settings = load_settings(auto_ml_root)
    db = Database(settings.db_path)
    db.init_schema()
    gateway = ModelGateway(db=db, cache_dir=settings.root / ".model_cache")
    return settings, db, gateway


def call_model(gateway, prompt: str, purpose: str, job_id: str, max_output_tokens: int = 3200):
    from seismo_research_workbench.model_gateway import ModelRequest

    response = gateway.complete_with_fallback(
        ModelRequest(
            provider_id="deepseek",
            prompt=prompt,
            purpose=purpose,
            job_id=job_id,
            max_output_tokens=max_output_tokens,
            use_cache=False,
            require_configured=True,
        )
    )
    return response


def evidence_pack() -> str:
    return "\n\n".join(
        [
            "## Current English Draft Excerpt\n" + read_text(ROOT / "docs" / "submission_draft_en.md", 12000),
            "## Current Literature Matrix\n" + read_text(ROOT / "docs" / "literature_matrix_zh.md", 9000),
            "## Result Tables\n" + read_text(ROOT / "docs" / "paper_result_tables_current.md", 6000),
            "## Reviewer Gap Audit\n" + read_text(ROOT / "docs" / "top_journal_reviewer_gap_audit_zh.md", 6000),
        ]
    )


def run(auto_ml_root: Path) -> dict:
    settings, _db, gateway = build_gateway(auto_ml_root)
    status = gateway.test_provider("deepseek")
    if not status.ok:
        raise RuntimeError(f"DeepSeek provider is not ready: {status.message}")

    pack = evidence_pack()
    lit_prompt = (
        "你是地震学、GNSS测地学与机器学习交叉方向的文献综述编辑。"
        "只能基于下面 evidence pack 给出建议，不要编造具体论文题名、DOI 或实验结果。"
        "请输出中文，结构包括：\n"
        "1. 当前 Related Work 的强项；\n"
        "2. 顶刊还会要求补齐的文献簇；\n"
        "3. 每个文献簇应回答的科学问题；\n"
        "4. 论文定位应如何收窄；\n"
        "5. 需要在正文中避免的过度主张。\n\n"
        "请保持紧凑，不超过 2200 个中文字。\n\n"
        + pack
    )
    polish_prompt = (
        "你是顶级地学/机器学习交叉期刊的严格审稿人兼论文润色编辑。"
        "只能基于下面 evidence pack 审阅，不要编造新结果。"
        "请输出中文，结构包括：\n"
        "1. 录用潜力判断；\n"
        "2. major concerns；\n"
        "3. minor concerns；\n"
        "4. 摘要、引言、方法、结果、讨论各节的具体改写建议；\n"
        "5. 投稿前必须补充的实验和图表；\n"
        "6. 一段可直接替换摘要最后两句的更审慎表述。\n\n"
        "请保持紧凑，不超过 3000 个中文字，最后必须完整给出替换句。\n\n"
        + pack
    )

    lit_response = call_model(gateway, lit_prompt, "sse_deepseek_literature_review", "sse_deepseek_literature", 2800)
    polish_response = call_model(gateway, polish_prompt, "sse_deepseek_manuscript_review", "sse_deepseek_polish", 3800)

    header_lit = (
        f"# DeepSeek 辅助文献定位审阅\n\n"
        f"日期：{date.today().isoformat()}\n\n"
        f"Provider: `{lit_response.provider}` / Model: `{lit_response.model}`\n\n"
        "> 本文档由 auto_ml 已配置的 DeepSeek provider 生成。它是论文修改建议，不是新的实验证据；所有论文事实仍以 SSE 仓库中的审计、训练和文献文件为准。\n\n"
    )
    header_polish = (
        f"# DeepSeek 辅助顶刊审稿与润色意见\n\n"
        f"日期：{date.today().isoformat()}\n\n"
        f"Provider: `{polish_response.provider}` / Model: `{polish_response.model}`\n\n"
        "> 本文档由 auto_ml 已配置的 DeepSeek provider 生成。它是外部审稿式意见，不替代人工审阅；不得把其中建议当作已完成实验。\n\n"
    )
    outputs = {
        "literature_review": ROOT / "docs" / "deepseek_literature_and_positioning_review_zh.md",
        "manuscript_review": ROOT / "docs" / "deepseek_manuscript_polish_review_zh.md",
    }
    write_text(outputs["literature_review"], header_lit + lit_response.text)
    write_text(outputs["manuscript_review"], header_polish + polish_response.text)

    project_reviews = auto_ml_root / "projects" / AUTO_PROJECT_ID / "reviews"
    write_text(project_reviews / outputs["literature_review"].name, header_lit + lit_response.text)
    write_text(project_reviews / outputs["manuscript_review"].name, header_polish + polish_response.text)

    return {
        "provider_status": {"provider": status.provider_id, "ok": status.ok, "message": status.message},
        "outputs": {name: str(path) for name, path in outputs.items()},
        "auto_ml_reviews": str(project_reviews),
        "auto_ml_root": str(settings.root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-ml-root", required=True)
    args = parser.parse_args()
    import json

    print(json.dumps(run(Path(args.auto_ml_root)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
