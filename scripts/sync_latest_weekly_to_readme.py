#!/usr/bin/env python3
"""Sync the latest weekly report's README block into README.md."""
from __future__ import annotations

import re
from pathlib import Path

README = Path("README.md")
WEEKLY_DIR = Path("weekly")
START = "<!-- WEEKLY_CHINESE_LLM_UPDATE:START -->"
END = "<!-- WEEKLY_CHINESE_LLM_UPDATE:END -->"


def main() -> int:
    reports = sorted(WEEKLY_DIR.glob("20??-??-??.md"))
    if not reports:
        print("No weekly reports found.")
        return 0

    latest = reports[-1]
    date = latest.stem
    report = latest.read_text(encoding="utf-8")
    match = re.search(re.escape(START) + r".*?" + re.escape(END), report, re.S)
    if not match:
        raise RuntimeError(f"{latest} does not contain a README update block")

    readme = README.read_text(encoding="utf-8")
    readme = re.sub(
        r"最近更新：\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日",
        f"最近更新：{int(date[:4])} 年 {int(date[5:7])} 月 {int(date[8:10])} 日",
        readme,
        count=1,
    )
    block_pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if block_pattern.search(readme):
        readme = block_pattern.sub(match.group(0), readme, count=1)
    else:
        marker = "---\n\n## 目录"
        readme = readme.replace(marker, f"---\n\n{match.group(0)}\n\n---\n\n## 目录", 1)

    README.write_text(readme, encoding="utf-8")
    print(f"Synced {latest} into README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
