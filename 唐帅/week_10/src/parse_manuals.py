"""
技术手册 Markdown 解析脚本：将原始 markdown 手册转换为结构化 JSON，供 RAG 使用。

设计目标：
  1. 直接利用 markdown 的标题、代码块、列表、引用、表格结构
  2. 清洗 PDF 转 markdown 后常见的转义噪声
  3. 保留溯源信息：来源文件、标题路径、行号
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_markdown"
PARSED_DIR = BASE_DIR / "data" / "parsed_markdown"
PARSED_DIR.mkdir(parents=True, exist_ok=True)


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^```([\w#+-]*)\s*$")
ORDERED_LIST_RE = re.compile(r"^\s*\d+\.\s+")
UNORDERED_LIST_RE = re.compile(r"^\s*[-*+]\s+")
BLOCKQUOTE_RE = re.compile(r"^\s*>\s?")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
MARKDOWN_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!])")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")


@dataclass
class ParsedBlock:
    block_type: str             # "title" | "text" | "code" | "list" | "quote" | "table" | "notice"
    content: str
    page_num: int
    section_path: list[str]
    source_file: str
    title_level: int = 0
    line_start: int = 0
    line_end: int = 0
    raw_table: list[list[str]] | None = field(default=None, repr=False)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = text.replace("\xa0", " ").replace("\u3000", " ")
    text = MARKDOWN_ESCAPE_RE.sub(r"\1", text)
    text = INLINE_LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    text = INLINE_CODE_RE.sub(lambda m: m.group(1), text)
    text = text.replace("**", "").replace("__", "")
    text = text.replace("\\|", "|")
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def clean_code_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def is_blank(line: str) -> bool:
    return not line.strip()


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    if stripped.startswith("```"):
        return False
    return stripped.count("|") >= 2


def split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [clean_text(cell) for cell in stripped.split("|")]


def normalize_table_rows(rows: list[list[str]]) -> list[list[str]]:
    width = max((len(row) for row in rows), default=0)
    normalized = []
    for row in rows:
        padded = list(row) + [""] * (width - len(row))
        normalized.append(padded)
    return normalized


def table_to_markdown(rows: list[list[str]]) -> str:
    rows = normalize_table_rows(rows)
    if not rows:
        return ""
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


class MarkdownManualParser:
    def __init__(self, md_path: Path):
        self.md_path = md_path
        self.blocks: list[ParsedBlock] = []
        self._section_stack: list[str] = []
        self._document_title: str | None = None

    def _update_section(self, title: str, level: int) -> None:
        level = max(1, min(level, 6))
        if level == 1:
            if self._document_title is None:
                self._document_title = title
                self._section_stack = [title]
            elif title == self._document_title:
                self._section_stack = [title]
            else:
                self._section_stack = [self._document_title, title]
            return

        if self._document_title and self._section_stack:
            base = [self._document_title]
            effective_level = level
            rest = self._section_stack[1:]
            rest = rest[: effective_level - 2] + [title]
            self._section_stack = base + rest
            return

        self._section_stack = self._section_stack[: level - 1] + [title]

    def _append_block(
        self,
        block_type: str,
        content: str,
        line_start: int,
        line_end: int,
        *,
        title_level: int = 0,
        raw_table: list[list[str]] | None = None,
    ) -> None:
        content = content.strip()
        if not content:
            return
        self.blocks.append(
            ParsedBlock(
                block_type=block_type,
                content=content,
                page_num=1,
                section_path=list(self._section_stack),
                source_file=self.md_path.name,
                title_level=title_level,
                line_start=line_start,
                line_end=line_end,
                raw_table=raw_table,
            )
        )

    def _consume_paragraph(self, lines: list[str], start_idx: int) -> tuple[int, str, int]:
        buffer = []
        idx = start_idx
        while idx < len(lines):
            raw = lines[idx]
            if is_blank(raw):
                break
            stripped = raw.strip()
            if (
                HEADING_RE.match(stripped)
                or FENCE_RE.match(stripped)
                or BLOCKQUOTE_RE.match(stripped)
                or ORDERED_LIST_RE.match(stripped)
                or UNORDERED_LIST_RE.match(stripped)
            ):
                break
            if is_table_line(stripped) and idx + 1 < len(lines) and TABLE_SEPARATOR_RE.match(lines[idx + 1].strip()):
                break
            buffer.append(clean_text(stripped))
            idx += 1
        return idx, " ".join(part for part in buffer if part), idx

    def _consume_code_block(self, lines: list[str], start_idx: int) -> tuple[int, str, int]:
        start_line = lines[start_idx].strip()
        match = FENCE_RE.match(start_line)
        language = (match.group(1) or "").strip()
        buffer = []
        idx = start_idx + 1
        while idx < len(lines):
            if lines[idx].strip().startswith("```"):
                idx += 1
                break
            buffer.append(lines[idx].rstrip("\n"))
            idx += 1
        code = clean_code_text("\n".join(buffer))
        if language:
            code = f"{language}\n{code}" if code else language
        return idx, code, idx

    def _consume_list_block(self, lines: list[str], start_idx: int) -> tuple[int, str, int]:
        buffer = []
        idx = start_idx
        while idx < len(lines):
            raw = lines[idx]
            stripped = raw.rstrip()
            if is_blank(stripped):
                look_ahead = idx + 1
                while look_ahead < len(lines) and is_blank(lines[look_ahead]):
                    look_ahead += 1
                if look_ahead < len(lines):
                    next_line = lines[look_ahead].rstrip()
                    if (
                        ORDERED_LIST_RE.match(next_line)
                        or UNORDERED_LIST_RE.match(next_line)
                        or next_line.startswith("    ")
                    ):
                        idx = look_ahead
                        continue
                break
            if not (ORDERED_LIST_RE.match(stripped) or UNORDERED_LIST_RE.match(stripped) or stripped.startswith("    ")):
                break
            cleaned = clean_text(stripped)
            if ORDERED_LIST_RE.match(stripped) or UNORDERED_LIST_RE.match(stripped):
                buffer.append(cleaned)
            else:
                if buffer:
                    buffer[-1] = f"{buffer[-1]}\n{cleaned}"
                else:
                    buffer.append(cleaned)
            idx += 1
        return idx, "\n".join(part for part in buffer if part), idx

    def _consume_quote_block(self, lines: list[str], start_idx: int) -> tuple[int, str, int]:
        buffer = []
        idx = start_idx
        while idx < len(lines):
            raw = lines[idx]
            if is_blank(raw):
                break
            if not BLOCKQUOTE_RE.match(raw):
                break
            cleaned = BLOCKQUOTE_RE.sub("", raw.strip(), count=1)
            buffer.append(clean_text(cleaned))
            idx += 1
        return idx, "\n".join(part for part in buffer if part), idx

    def _consume_table_block(self, lines: list[str], start_idx: int) -> tuple[int, str, list[list[str]], int]:
        rows = [split_table_row(lines[start_idx])]
        idx = start_idx + 2
        while idx < len(lines):
            raw = lines[idx].strip()
            if is_blank(raw) or not is_table_line(raw):
                break
            rows.append(split_table_row(raw))
            idx += 1
        rows = [row for row in rows if any(cell for cell in row)]
        markdown = table_to_markdown(rows)
        return idx, markdown, rows, idx

    def parse(self) -> list[ParsedBlock]:
        logger.info("开始解析: %s", self.md_path.name)

        text = self.md_path.read_text(encoding="utf-8")
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        idx = 0
        while idx < len(lines):
            raw = lines[idx]
            stripped = raw.strip()
            line_no = idx + 1

            if is_blank(raw):
                idx += 1
                continue

            heading_match = HEADING_RE.match(stripped)
            if heading_match:
                level = len(heading_match.group(1))
                title = clean_text(heading_match.group(2))
                self._update_section(title, level)
                self._append_block("title", title, line_no, line_no, title_level=level)
                idx += 1
                continue

            if FENCE_RE.match(stripped):
                next_idx, content, end_idx = self._consume_code_block(lines, idx)
                self._append_block("code", content, line_no, end_idx)
                idx = next_idx
                continue

            if is_table_line(stripped) and idx + 1 < len(lines) and TABLE_SEPARATOR_RE.match(lines[idx + 1].strip()):
                next_idx, markdown, rows, end_idx = self._consume_table_block(lines, idx)
                self._append_block("table", markdown, line_no, end_idx, raw_table=rows)
                idx = next_idx
                continue

            if ORDERED_LIST_RE.match(stripped) or UNORDERED_LIST_RE.match(stripped):
                next_idx, content, end_idx = self._consume_list_block(lines, idx)
                self._append_block("list", content, line_no, end_idx)
                idx = next_idx
                continue

            if BLOCKQUOTE_RE.match(stripped):
                next_idx, content, end_idx = self._consume_quote_block(lines, idx)
                self._append_block("quote", content, line_no, end_idx)
                idx = next_idx
                continue

            if stripped == "---":
                idx += 1
                continue

            next_idx, content, end_idx = self._consume_paragraph(lines, idx)
            self._append_block("text", content, line_no, end_idx)
            idx = next_idx if next_idx > idx else idx + 1

        if not self.blocks:
            self._append_block("notice", "[未提取到有效 markdown 内容]", 1, 1)

        logger.info("  解析完成: %d 个块", len(self.blocks))
        return self.blocks

    def save(self, output_dir: Path = PARSED_DIR) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{self.md_path.stem}.json"
        payload = {
            "source": str(self.md_path),
            "source_file": self.md_path.name,
            "source_type": "markdown",
            "parser": "MarkdownManualParser",
            "blocks": [asdict(block) for block in self.blocks],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("  已保存 -> %s", out_path)
        return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="解析技术手册 markdown，输出结构化 JSON")
    parser.add_argument("--md", type=str, default="", help="只解析单个 markdown 文件")
    parser.add_argument("--input-dir", type=str, default=str(RAW_DIR), help="原始 markdown 目录")
    parser.add_argument("--output-dir", type=str, default=str(PARSED_DIR), help="解析结果目录")
    return parser.parse_args()


def iter_markdowns(md_arg: str, input_dir: Path) -> list[Path]:
    if md_arg:
        md_path = Path(md_arg)
        if not md_path.is_absolute():
            md_path = input_dir / md_path
        return [md_path]
    return sorted(input_dir.glob("*.md"))


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error("输入目录不存在: %s", input_dir)
        return

    md_paths = iter_markdowns(args.md, input_dir)
    if not md_paths:
        logger.error("没有找到任何 markdown 文件: %s", input_dir)
        return

    for md_path in md_paths:
        if not md_path.exists():
            logger.warning("文件不存在，跳过: %s", md_path)
            continue

        parser = MarkdownManualParser(md_path)
        parser.parse()
        parser.save(output_dir)

    logger.info("全部解析完成，结果目录: %s", output_dir)


if __name__ == "__main__":
    main()
