#!/usr/bin/env python3
"""Weekly updater for ChatGPT-Chinese-Tutorial.

Collect recent high-value ChatGPT / LLM / Agent / RAG / MCP / post-training /
multimodal learning resources from the broader web. Candidate resources may be
Chinese or English, and may come from websites, blogs, docs, papers, courses,
repositories, model hubs, newsletters, or community posts.

The final README section and weekly archive are written in Chinese for readers.

Optional repo secrets:
- SERPER_API_KEY: enables broad web search beyond GitHub / Hugging Face / arXiv.
- OPENAI_API_KEY: reserved for higher-quality Chinese summarization/ranking.
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

# Broad discovery topics. Do not restrict to Chinese content; the output remains Chinese.
DISCOVERY_TOPICS = [
    "ChatGPT tutorial LLM learning resources",
    "large language model course guide tutorial",
    "AI agent tool use MCP workflow tutorial",
    "RAG retrieval augmented generation search tutorial",
    "LLM post-training SFT DPO GRPO RLHF guide",
    "multimodal LLM VLM tutorial benchmark",
    "prompt engineering context engineering guide",
    "open source LLM reasoning coding agent",
    "ChatGPT 中文 教程 大模型 学习资料",
    "大模型 Agent RAG MCP 后训练 多模态 学习资料",
]

WEB_QUERIES = [
    "latest LLM learning resources tutorial course blog paper",
    "best AI agent MCP tool use tutorial guide",
    "latest RAG LLM retrieval tutorial benchmark blog",
    "LLM post training SFT DPO GRPO RLHF tutorial paper",
    "multimodal LLM VLM tutorial benchmark latest",
    "prompt engineering context engineering latest guide",
    "ChatGPT 中文 大模型 最新 教程 学习资料",
    "大模型 Agent RAG MCP 后训练 最新 学习资料",
]

CATEGORIES = [
    ("AI Agent / 工具调用", ["agent", "tool", "mcp", "a2a", "workflow", "computer use", "智能体", "工具调用"]),
    ("RAG / AI 搜索", ["rag", "retrieval", "search", "vector", "ranking", "检索", "搜索", "知识库"]),
    ("后训练 / 强化学习", ["sft", "dpo", "grpo", "ppo", "rlhf", "rlaif", "post-training", "finetune", "fine-tuning", "微调", "后训练", "强化学习"]),
    ("Prompt / 上下文工程", ["prompt", "context", "context engineering", "提示词", "上下文"]),
    ("多模态", ["multimodal", "vision-language", "vlm", "video", "audio", "omni", "多模态", "视觉", "视频", "音频"]),
    ("开源模型 / 模型平台", ["qwen", "deepseek", "glm", "llama", "kimi", "model", "hugging face", "模型"]),
    ("课程 / 文档 / 博客", ["course", "tutorial", "guide", "handbook", "docs", "blog", "newsletter", "课程", "教程", "指南", "文档"]),
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


def zh_summary(text: str, fallback: str = "近期值得关注的学习资源。") -> str:
    """Lightweight Chinese expression fallback without requiring an LLM."""
    text = (text or "").strip()
    if not text:
        return fallback
    if re.search(r"[\u4e00-\u9fff]", text):
        return text[:220] + ("..." if len(text) > 220 else "")
    return "英文资源，建议关注：" + text[:180] + ("..." if len(text) > 180 else "")


def collect_github(since: str) -> list[Item]:
    token = os.getenv("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "weekly-llm-updater"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    out: list[Item] = []
    for topic in DISCOVERY_TOPICS:
        query = f"{topic} pushed:>={since}"
        url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
            {"q": query, "sort": "updated", "order": "desc", "per_page": 10}
        )
        try:
            data = http_json(url, headers=headers)
            for repo in data.get("items", []):
                title = repo.get("full_name") or repo.get("name") or "GitHub repository"
                desc = (repo.get("description") or "Recently active open-source project.").strip()
                stars = int(repo.get("stargazers_count") or 0)
                updated = (repo.get("pushed_at") or repo.get("updated_at") or "")[:10]
                text = f"{title} {desc}"
                score = 22 + min(stars, 20000) / 800
                if re.search(r"chatgpt|llm|agent|rag|mcp|multimodal|post-training|tutorial|course|awesome|大模型|教程", text, re.I):
                    score += 18
                out.append(Item(title, repo["html_url"], "GitHub", zh_summary(desc), category_of(text), score, f"⭐ {stars}", updated))
        except Exception as e:
            print(f"skip GitHub topic {topic}: {e}")
        time.sleep(1)
    return out


def collect_huggingface() -> list[Item]:
    out: list[Item] = []
    for q in ["llm", "agent", "rag", "multimodal", "post-training", "qwen", "deepseek", "chinese llm"]:
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
                score = 10 + min(downloads, 200000) / 10000 + likes / 20
                out.append(Item(model_id, f"https://huggingface.co/{model_id}", "Hugging Face", summary, category_of(model_id + tags), score, f"likes {likes}", updated))
        except Exception as e:
            print(f"skip Hugging Face {q}: {e}")
        time.sleep(0.5)
    return out


def collect_arxiv() -> list[Item]:
    query = 'all:("large language model" OR "LLM" OR "AI agent" OR "RAG" OR "tool use" OR "post-training" OR "multimodal")'
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "sortBy": "submittedDate", "sortOrder": "descending", "max_results": 30}
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
            out.append(Item(title, link, "arXiv", zh_summary(summary), category_of(title + summary), 15, "paper", published))
    except Exception as e:
        print(f"skip arXiv: {e}")
    return out


def collect_web() -> list[Item]:
    """Broad web search. This is the main source for truly full-web discovery."""
    key = os.getenv("SERPER_API_KEY", "")
    if not key:
        return []
    out: list[Item] = []
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    for topic in WEB_QUERIES:
        payload = json.dumps({"q": topic, "num": 10, "hl": "zh-cn"}).encode()
        try:
            data = http_json("https://google.serper.dev/search", headers=headers, data=payload)
            for row in data.get("organic", []) or []:
                title, link = row.get("title", ""), row.get("link", "")
                snippet = row.get("snippet", "")
                if link and title:
                    out.append(Item(title, link, "Web", zh_summary(snippet), category_of(title + snippet), 35))
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
        if any(x in text for x in ["教程", "指南", "课程", "awesome", "tutorial", "course", "handbook", "guide", "docs", "blog"]):
            it.score += 24
        if any(x in text for x in ["chatgpt", "llm", "large language model", "大模型"]):
            it.score += 12
        if any(x in text for x in ["agent", "rag", "mcp", "grpo", "dpo", "rlhf", "tool", "post-training", "后训练", "智能体"]):
            it.score += 12
        if it.source == "Web":
            it.score += 8
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
        f"> 自动生成时间：{date}。每周从全网筛选近期 ChatGPT / LLM / Agent / RAG / MCP / 后训练 / 多模态等高价值学习资源；候选资料不限中文，英文资料也会纳入，最终统一用中文表达学习价值。",
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
    lines = [
        f"# ChatGPT / LLM 每周学习精选（{date}）",
        "",
        "由 GitHub Actions 自动生成。候选来源面向全网，不限中文或英文，最终以中文总结学习价值。",
        "",
    ]
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
