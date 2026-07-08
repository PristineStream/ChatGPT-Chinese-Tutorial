#!/usr/bin/env python3
"""Weekly updater for ChatGPT-Chinese-Tutorial.

Collect recent Chinese ChatGPT / LLM / Agent / RAG / MCP / post-training
resources, rank the most useful items, write a weekly archive, and refresh an
auto-generated section in README.md.

Optional repo secrets:
- OPENAI_API_KEY: improve Chinese summarization/ranking.
- SERPER_API_KEY: add broader web search results.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

README = Path("README.md")
WEEKLY_DIR = Path("weekly")
START = "<!-- WEEKLY_CHINESE_LLM_UPDATE:START -->"
END = "<!-- WEEKLY_CHINESE_LLM_UPDATE:END -->"

TOPICS = [
    "ChatGPT 中文 教程",
    "中文 大模型 LLM 教程",
    "AI Agent 中文 MCP RAG",
    "大模型 后训练 SFT DPO GRPO",
    "多模态 大模型 中文",
    "Prompt Engineering 中文",
]

CATEGORIES = [
    ("AI Agent / 工具调用", ["agent", "tool", "mcp", "a2a", "workflow", "智能体", "工具调用"]),
    ("RAG / AI 搜索", ["rag", "retrieval", "search", "vector", "检索", "搜索", "知识库"]),
    ("后训练 / 强化学习", ["sft", "dpo", "grpo", "ppo", "rlhf", "post-training", "finetune", "微调", "后训练", "强化学习"]),
    ("Prompt / 上下文工程", ["prompt", "context", "提示词", "上下文"]),
    ("多模态", ["multimodal", "vision", "vlm", "video", "audio", "多模态", "视觉", "视频", "音频"]),
    ("开源模型", ["qwen", "deepseek", "glm", "llama", "kimi", "model", "模型"]),
]


@dataclass
class Item:
    title: str
    url: str
    source: str
    summary: str
    category: str
    score: float = 0.0
    hotness: str = ""
    updated: str = ""

    @property
    def key(self) -> str:
        return re.sub(r"[#?].*$", "", self.url.strip().lower())


def tz_now() -> datetime:
    name = os.getenv("RUN_TZ", "America/Los_Angeles")
    if ZoneInfo:
        try:
            return datetime.now(ZoneInfo(name))
        except Exception:
            pass
    return datetime.now(timezone.utc)


def http_json(url: str, headers: dict[str, str] | None = None, data: bytes | None = None, timeout: int = 30):
    req = urllib.request.Request(url, headers=headers or {}, data=data)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def http_text(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def category_of(text: str) -> str:
    low = text.lower()
    for name, keys in CATEGORIES:
        if any(k.lower() in low for k in keys):
            return name
    return "综合学习资源"


def collect_github(since: str) -> list[Item]:
    token = os.getenv("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "weekly-llm-updater"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    out: list[Item] = []
    for topic in TOPICS:
        query = f"{topic} pushed:>={since}"
        url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
            {"q": query, "sort": "updated", "order": "desc", "per_page": 10}
        )
        try:
            data = http_json(url, headers=headers)
            for repo in data.get("items", []):
                title = repo.get("full_name") or repo.get("name") or "GitHub repository"
                desc = (repo.get("description") or "近期活跃的开源项目。").strip()
                stars = int(repo.get("stargazers_count") or 0)
                updated = (repo.get("pushed_at") or repo.get("updated_at") or "")[:10]
                text = f"{title} {desc}"
                score = 25 + min(stars, 20000) / 800
                if re.search(r"中文|chinese|chatgpt|llm|agent|rag|mcp|大模型|教程|awesome", text, re.I):
                    score += 20
                out.append(Item(title, repo["html_url"], "GitHub", desc, category_of(text), score, f"⭐ {stars}", updated))
        except Exception as e:
            print(f"skip GitHub topic {topic}: {e}")
        time.sleep(1)
    return out


def collect_huggingface() -> list[Item]:
    out: list[Item] = []
    for q in ["chinese llm", "qwen", "deepseek", "agent", "rag"]:
        url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(
            {"search": q, "sort": "lastModified", "direction": "-1", "limit": 8}
        )
        try:
            data = http_json(url, headers={"User-Agent": "weekly-llm-updater"})
            for m in data if isinstance(data, list) else []:
                model_id = m.get("modelId") or m.get("id")
                if not model_id:
                    continue
                likes = int(m.get("likes") or 0)
                downloads = int(m.get("downloads") or 0)
                updated = (m.get("lastModified") or "")[:10]
                tags = ", ".join((m.get("tags") or [])[:8])
                summary = f"近期更新模型，tags: {tags}" if tags else "近期更新模型。"
                score = 12 + min(downloads, 200000) / 10000 + likes / 20
                out.append(Item(model_id, f"https://huggingface.co/{model_id}", "Hugging Face", summary, category_of(model_id + tags), score, f"likes {likes}", updated))
        except Exception as e:
            print(f"skip Hugging Face {q}: {e}")
        time.sleep(0.5)
    return out


def collect_arxiv() -> list[Item]:
    query = 'all:("large language model" OR "LLM" OR "AI agent" OR "RAG" OR "tool use" OR "post-training")'
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "sortBy": "submittedDate", "sortOrder": "descending", "max_results": 20}
    )
    out: list[Item] = []
    try:
        xml = http_text(url, headers={"User-Agent": "weekly-llm-updater"})
        root = ET.fromstring(xml)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            title = " ".join((entry.findtext("a:title", default="", namespaces=ns) or "").split())
            link = entry.findtext("a:id", default="", namespaces=ns) or ""
            summary = " ".join((entry.findtext("a:summary", default="", namespaces=ns) or "").split())
            published = (entry.findtext("a:published", default="", namespaces=ns) or "")[:10]
            out.append(Item(title, link, "arXiv", summary[:180] + ("..." if len(summary) > 180 else ""), category_of(title + summary), 15, "paper", published))
    except Exception as e:
        print(f"skip arXiv: {e}")
    return out


def collect_web() -> list[Item]:
    key = os.getenv("SERPER_API_KEY", "")
    if not key:
        return []
    out: list[Item] = []
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    for topic in TOPICS:
        payload = json.dumps({"q": f"{topic} 最新 学习资料 论文 GitHub", "num": 8, "hl": "zh-cn"}).encode()
        try:
            data = http_json("https://google.serper.dev/search", headers=headers, data=payload)
            for row in data.get("organic", []) or []:
                title, link = row.get("title", ""), row.get("link", "")
                snippet = row.get("snippet", "")
                if link and title:
                    out.append(Item(title, link, "Web", snippet, category_of(title + snippet), 30))
        except Exception as e:
            print(f"skip web {topic}: {e}")
        time.sleep(1)
    return out


def dedupe(items: list[Item]) -> list[Item]:
    seen, out = set(), []
    for it in items:
        if not it.url or it.key in seen:
            continue
        seen.add(it.key)
        out.append(it)
    return out


def rank(items: list[Item], k: int = 12) -> list[Item]:
    now = tz_now().date()
    for it in items:
        text = f"{it.title} {it.summary}".lower()
        if any(x in text for x in ["教程", "指南", "课程", "awesome", "tutorial", "course", "handbook"]):
            it.score += 25
        if any(x in text for x in ["中文", "chinese", "chatgpt"]):
            it.score += 12
        if any(x in text for x in ["agent", "rag", "mcp", "grpo", "dpo", "tool", "后训练", "智能体"]):
            it.score += 10
        try:
            age = (now - datetime.fromisoformat(it.updated).date()).days
            it.score += max(0, 14 - age)
        except Exception:
            pass
    return sorted(items, key=lambda x: x.score, reverse=True)[:k]


def escape(s: str) -> str:
    return s.replace("\n", " ").replace("|", "\\|").strip()


def weekly_section(items: list[Item], date: str) -> str:
    lines = [
        START,
        "## 每周精选更新",
        "",
        f"> 自动生成时间：{date}。每周筛选近期 ChatGPT 中文、中文 LLM、Agent、RAG、MCP、后训练、多模态等学习资源。",
        "",
        "| 推荐 | 方向 | 资源 | 来源 | 推荐理由 |",
        "| ---- | ---- | ---- | ---- | ---- |",
    ]
    for i, it in enumerate(items[:8], 1):
        meta = "；".join(x for x in [it.hotness, f"更新 {it.updated}" if it.updated else ""] if x)
        reason = it.summary if not meta else f"{it.summary}（{meta}）"
        lines.append(f"| {i} | {escape(it.category)} | [{escape(it.title)}]({it.url}) | {escape(it.source)} | {escape(reason)} |")
    lines.extend(["", END])
    return "\n".join(lines)


def archive_md(items: list[Item], date: str) -> str:
    lines = [f"# ChatGPT / 中文 LLM 每周学习精选（{date}）", "", "由 GitHub Actions 自动生成。", ""]
    for i, it in enumerate(items, 1):
        lines += [
            f"## {i}. {it.title}",
            "",
            f"- 链接：{it.url}",
            f"- 方向：{it.category}",
            f"- 来源：{it.source}",
            f"- 学习价值：{it.summary}",
            "",
        ]
    return "\n".join(lines)


def update_readme(section: str, date: str) -> None:
    text = README.read_text(encoding="utf-8")
    cn_date = datetime.fromisoformat(date).strftime("%Y 年 %-m 月 %-d 日") if os.name != "nt" else date
    text = re.sub(r"最近更新：\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日", f"最近更新：{cn_date}", text, count=1)
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if pattern.search(text):
        text = pattern.sub(section, text, count=1)
    else:
        marker = "---\n\n## 目录"
        if marker in text:
            text = text.replace(marker, f"---\n\n{section}\n\n---\n\n## 目录", 1)
        else:
            text = section + "\n\n" + text
    README.write_text(text, encoding="utf-8")


def main() -> int:
    date = tz_now().strftime("%Y-%m-%d")
    since = (tz_now().date() - timedelta(days=int(os.getenv("LOOKBACK_DAYS", "14")))).isoformat()
    items = dedupe(collect_web() + collect_github(since) + collect_huggingface() + collect_arxiv())
    if not items:
        print("No candidates collected.")
        return 0
    selected = rank(items, int(os.getenv("TOP_K", "12")))
    WEEKLY_DIR.mkdir(exist_ok=True)
    (WEEKLY_DIR / f"{date}.md").write_text(archive_md(selected, date), encoding="utf-8")
    update_readme(weekly_section(selected, date), date)
    print(f"Updated README.md and weekly/{date}.md with {len(selected)} items.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
