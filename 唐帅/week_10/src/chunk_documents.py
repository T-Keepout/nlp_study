"""
Markdown 手册分块脚本：对解析后的手册 JSON 做 chunk，供向量检索使用。

输出格式与课件版本尽量保持一致：
  - chunk_id
  - content
  - metadata

适配点：
  1. 输入来自 parsed_markdown/*.json
  2. 主要处理 text / quote / list / code / table / title 六类 block
  3. chunk 时保留 section_path、line_start/line_end、source_file 等溯源信息
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterator


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).parent.parent
PARSED_DIR = BASE_DIR / "data" / "parsed_markdown"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_STRATEGY = "semantic"


def slugify_filename(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "document"


def format_chunk_content(
    content: str,
    *,
    source_file: str,
    section_path: list[str],
    block_types: list[str],
) -> str:
    prefix_lines = [f"文档: {source_file}"]
    if section_path:
        prefix_lines.append(f"章节: {' > '.join(section_path)}")
    if block_types:
        prefix_lines.append(f"类型: {', '.join(block_types)}")
    prefix = "\n".join(prefix_lines)
    return f"{prefix}\n\n{content.strip()}"


def build_chunk_id(source_file: str, idx: int) -> str:
    return f"{slugify_filename(source_file)}_{idx:05d}"


def chunk_fixed(text: str, chunk_size: int = 700, overlap: int = 80) -> Iterator[str]:
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        if end >= len(text):
            break
        start += max(1, chunk_size - overlap)


def split_large_text(content: str, max_chunk_size: int, overlap: int = 80) -> list[str]:
    if len(content) <= max_chunk_size:
        return [content]

    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    if len(paragraphs) <= 1:
        return list(chunk_fixed(content, chunk_size=max_chunk_size, overlap=overlap))

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for para in paragraphs:
        para_len = len(para)
        extra = 1 if buffer else 0
        if buffer and buffer_len + para_len + extra > max_chunk_size:
            chunks.append("\n".join(buffer))
            buffer = [para]
            buffer_len = para_len
        else:
            buffer.append(para)
            buffer_len += para_len + extra

    if buffer:
        chunks.append("\n".join(buffer))

    if any(len(chunk) > max_chunk_size for chunk in chunks):
        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                final_chunks.extend(chunk_fixed(chunk, chunk_size=max_chunk_size, overlap=overlap))
            else:
                final_chunks.append(chunk)
        return final_chunks

    return chunks


def flush_semantic_buffer(
    buffer_blocks: list[dict],
    *,
    min_chunk_size: int,
) -> dict | None:
    if not buffer_blocks:
        return None

    section_path = buffer_blocks[0].get("section_path", [])
    source_file = buffer_blocks[0].get("source_file", "")
    content = "\n\n".join(block["content"].strip() for block in buffer_blocks if block["content"].strip())
    if not content or len(content) < min_chunk_size:
        return None

    block_types = sorted({block["block_type"] for block in buffer_blocks})
    metadata = {
        "page_num": buffer_blocks[0].get("page_num", 1),
        "section": " > ".join(section_path) if section_path else "",
        "section_path": section_path,
        "block_types": block_types,
        "line_start": min(block.get("line_start", 0) for block in buffer_blocks),
        "line_end": max(block.get("line_end", 0) for block in buffer_blocks),
        "source_file": source_file,
        "source_type": "markdown",
    }
    formatted_content = format_chunk_content(
        content,
        source_file=source_file,
        section_path=section_path,
        block_types=block_types,
    )
    return {"content": formatted_content, "metadata": metadata}


def chunk_semantic(
    blocks: list[dict],
    *,
    max_chunk_size: int = 900,
    min_chunk_size: int = 80,
    code_chunk_size: int = 1200,
) -> Iterator[dict]:
    buffer_blocks: list[dict] = []
    buffer_len = 0

    def flush_buffer() -> dict | None:
        nonlocal buffer_blocks, buffer_len
        result = flush_semantic_buffer(buffer_blocks, min_chunk_size=min_chunk_size)
        buffer_blocks = []
        buffer_len = 0
        return result

    for block in blocks:
        btype = block.get("block_type", "text")
        content = block.get("content", "").strip()
        if not content:
            continue

        if btype == "title":
            result = flush_buffer()
            if result:
                yield result
            continue

        if btype in {"code", "table"}:
            result = flush_buffer()
            if result:
                yield result

            section_path = block.get("section_path", [])
            source_file = block.get("source_file", "")
            raw_parts = split_large_text(
                content,
                max_chunk_size=code_chunk_size if btype == "code" else max_chunk_size,
                overlap=80,
            )
            for part in raw_parts:
                if len(part.strip()) < min_chunk_size and len(raw_parts) > 1:
                    continue
                yield {
                    "content": format_chunk_content(
                        part,
                        source_file=source_file,
                        section_path=section_path,
                        block_types=[btype],
                    ),
                    "metadata": {
                        "page_num": block.get("page_num", 1),
                        "section": " > ".join(section_path) if section_path else "",
                        "section_path": section_path,
                        "block_types": [btype],
                        "line_start": block.get("line_start", 0),
                        "line_end": block.get("line_end", 0),
                        "source_file": source_file,
                        "source_type": "markdown",
                    },
                }
            continue

        projected_len = buffer_len + len(content) + (2 if buffer_blocks else 0)
        same_section = not buffer_blocks or buffer_blocks[0].get("section_path", []) == block.get("section_path", [])

        if buffer_blocks and ((not same_section) or projected_len > max_chunk_size):
            result = flush_buffer()
            if result:
                yield result

        buffer_blocks.append(block)
        buffer_len += len(content) + (2 if len(buffer_blocks) > 1 else 0)

    result = flush_buffer()
    if result:
        yield result


def chunk_hierarchical(
    blocks: list[dict],
    *,
    parent_size: int = 1800,
    child_size: int = 500,
    overlap: int = 80,
) -> Iterator[dict]:
    semantic_chunks = list(chunk_semantic(blocks, max_chunk_size=parent_size, min_chunk_size=80, code_chunk_size=parent_size))
    for parent_idx, parent_chunk in enumerate(semantic_chunks):
        parent_id = f"parent_{parent_idx:05d}"
        parent_content = parent_chunk["content"]
        child_parts = split_large_text(parent_content, max_chunk_size=child_size, overlap=overlap)

        for child_content in child_parts:
            metadata = dict(parent_chunk["metadata"])
            metadata["parent_id"] = parent_id
            metadata["parent_content"] = parent_content
            yield {"content": child_content, "metadata": metadata}


def process_file(
    parsed_path: Path,
    *,
    strategy: str = DEFAULT_STRATEGY,
    max_chunk_size: int = 900,
    min_chunk_size: int = 80,
) -> list[dict]:
    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)

    blocks = data.get("blocks", [])
    source_file = data.get("source_file", parsed_path.name)
    logger.info("分块 %s  策略=%s  blocks=%d", parsed_path.name, strategy, len(blocks))

    raw_chunks: list[dict] = []

    if strategy == "fixed":
        full_text = "\n\n".join(block.get("content", "") for block in blocks if block.get("block_type") != "title")
        base_content = format_chunk_content(
            full_text,
            source_file=source_file,
            section_path=[],
            block_types=["text"],
        )
        for text_chunk in chunk_fixed(base_content, chunk_size=max_chunk_size, overlap=80):
            raw_chunks.append(
                {
                    "content": text_chunk,
                    "metadata": {
                        "page_num": 1,
                        "section": "",
                        "section_path": [],
                        "block_types": ["text"],
                        "line_start": 0,
                        "line_end": 0,
                        "source_file": source_file,
                        "source_type": "markdown",
                    },
                }
            )
    elif strategy == "semantic":
        raw_chunks.extend(
            chunk_semantic(
                blocks,
                max_chunk_size=max_chunk_size,
                min_chunk_size=min_chunk_size,
            )
        )
    elif strategy == "hierarchical":
        raw_chunks.extend(chunk_hierarchical(blocks))
    else:
        raise ValueError(f"未知策略: {strategy}")

    result: list[dict] = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_id = build_chunk_id(source_file, idx)
        metadata = dict(chunk["metadata"])
        metadata["strategy"] = strategy
        metadata["chunk_index"] = idx
        result.append(
            {
                "chunk_id": chunk_id,
                "content": chunk["content"],
                "metadata": metadata,
            }
        )

    out_path = CHUNKS_DIR / f"{parsed_path.stem}_{strategy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("  -> %d 个 chunk，已保存 %s", len(result), out_path.name)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 parsed_markdown 文档做 chunk")
    parser.add_argument("--input-dir", type=str, default=str(PARSED_DIR), help="解析结果目录")
    parser.add_argument("--output-dir", type=str, default=str(CHUNKS_DIR), help="chunk 输出目录")
    parser.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY, choices=["fixed", "semantic", "hierarchical"])
    parser.add_argument("--file", type=str, default="", help="只处理单个解析结果 JSON")
    parser.add_argument("--max-chunk-size", type=int, default=900, help="语义分块的最大字符数")
    parser.add_argument("--min-chunk-size", type=int, default=80, help="语义分块的最小字符数")
    return parser.parse_args()


def iter_parsed_files(input_dir: Path, file_arg: str) -> list[Path]:
    if file_arg:
        file_path = Path(file_arg)
        if not file_path.is_absolute():
            file_path = input_dir / file_path
        return [file_path]
    return sorted(input_dir.glob("*.json"))


def main() -> None:
    global CHUNKS_DIR

    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR = output_dir

    if not input_dir.exists():
        logger.error("输入目录不存在: %s", input_dir)
        return

    parsed_files = iter_parsed_files(input_dir, args.file)
    if not parsed_files:
        logger.error("没有找到解析结果，请先运行 parse_manuals.py")
        return

    all_chunks: list[dict] = []
    for path in parsed_files:
        if not path.exists():
            logger.warning("文件不存在，跳过: %s", path)
            continue
        chunks = process_file(
            path,
            strategy=args.strategy,
            max_chunk_size=args.max_chunk_size,
            min_chunk_size=args.min_chunk_size,
        )
        all_chunks.extend(chunks)

    combined_path = output_dir / f"all_{args.strategy}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    avg_len = sum(len(chunk["content"]) for chunk in all_chunks) / max(len(all_chunks), 1)
    logger.info("\n合并完成：共 %d 个 chunk -> %s", len(all_chunks), combined_path)
    logger.info("平均 chunk 长度: %.0f 字符", avg_len)

    code_count = sum(1 for chunk in all_chunks if "code" in chunk["metadata"].get("block_types", []))
    table_count = sum(1 for chunk in all_chunks if "table" in chunk["metadata"].get("block_types", []))
    logger.info("其中代码块: %d  表格块: %d", code_count, table_count)


if __name__ == "__main__":
    main()
