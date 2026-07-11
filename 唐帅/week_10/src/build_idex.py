"""
Markdown 手册向量索引构建脚本。

Embedding 方案：阿里云 DashScope text-embedding-v3
向量库：FAISS（IndexFlatIP，归一化后等价于余弦相似度）

依赖：
  pip install faiss-cpu openai numpy
  设置环境变量 DASHSCOPE_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from openai import OpenAI


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024
BATCH_SIZE = 10
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "请设置环境变量 DASHSCOPE_API_KEY\n"
            "  Windows: set DASHSCOPE_API_KEY=sk-xxx\n"
            "  Linux/Mac: export DASHSCOPE_API_KEY=sk-xxx"
        )
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


def embed_texts(client: OpenAI, texts: list[str], show_progress: bool = True) -> np.ndarray:
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1

        if show_progress and (batch_idx == 1 or batch_idx % 100 == 0):
            logger.info("  Embedding 进度: %d/%d 批", batch_idx, total_batches)

        for attempt in range(3):
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                    dimensions=EMBED_DIM,
                )
                all_embeddings.extend([item.embedding for item in resp.data])
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                logger.warning("  第%d次失败，重试: %s", attempt + 1, exc)
                time.sleep(2 ** attempt)

    embeddings = np.array(all_embeddings, dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    embeddings = embeddings / norms
    return embeddings


def build_faiss_index(chunks: list[dict], client: OpenAI, output_dir: Path):
    import faiss

    logger.info("开始计算 %d 条 chunk 的 embedding...", len(chunks))
    texts = [chunk["content"] for chunk in chunks]
    embeddings = embed_texts(client, texts)

    logger.info("构建 FAISS 索引，维度=%d...", EMBED_DIM)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    logger.info("索引构建完成，共 %d 条向量", index.ntotal)

    index_path = output_dir / "faiss_index.bin"
    meta_path = output_dir / "faiss_meta.json"

    faiss.write_index(index, str(index_path))
    logger.info("FAISS 索引已保存 -> %s", index_path)

    meta_list = [
        {
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "page_num": chunk["metadata"].get("page_num", 1),
            "section": chunk["metadata"].get("section", ""),
            "section_path": chunk["metadata"].get("section_path", []),
            "block_types": chunk["metadata"].get("block_types", []),
            "line_start": chunk["metadata"].get("line_start", 0),
            "line_end": chunk["metadata"].get("line_end", 0),
            "source_file": chunk["metadata"].get("source_file", ""),
            "source_type": chunk["metadata"].get("source_type", ""),
            "strategy": chunk["metadata"].get("strategy", ""),
            "chunk_index": chunk["metadata"].get("chunk_index", -1),
            "parent_content": chunk["metadata"].get("parent_content", ""),
            "parent_id": chunk["metadata"].get("parent_id", ""),
        }
        for chunk in chunks
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    logger.info("元数据已保存 -> %s", meta_path)

    return index, meta_list


def build_chroma_index(chunks: list[dict], client: OpenAI, output_dir: Path):
    try:
        import chromadb
    except ImportError:
        logger.error("请先安装 chromadb: pip install chromadb")
        return

    chroma_dir = output_dir / "chroma"
    client_db = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client_db.get_or_create_collection(
        name="technical_manuals",
        metadata={"hnsw:space": "cosine"},
    )

    logger.info("向 ChromaDB 写入 %d 条 chunk...", len(chunks))
    texts = [chunk["content"] for chunk in chunks]
    embeddings = embed_texts(client, texts)

    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        ids = [chunk["chunk_id"] for chunk in batch]
        docs = [chunk["content"] for chunk in batch]
        embs = embeddings[i : i + 100].tolist()
        metas = []
        for chunk in batch:
            meta = dict(chunk["metadata"])
            meta["block_types"] = ",".join(meta.get("block_types") or [])
            meta["section_path"] = " > ".join(meta.get("section_path") or [])
            meta.pop("parent_content", None)
            metas.append(meta)
        collection.add(documents=docs, embeddings=embs, ids=ids, metadatas=metas)
        logger.info("  已写入 %d/%d", min(i + 100, len(chunks)), len(chunks))

    logger.info("ChromaDB 写入完成，共 %d 条", collection.count())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为技术手册 chunks 构建向量索引")
    parser.add_argument("--strategy", type=str, default="semantic", choices=["fixed", "semantic", "hierarchical"])
    parser.add_argument("--chunks-file", type=str, default="", help="显式指定 chunks 文件")
    parser.add_argument("--output-dir", type=str, default=str(VECTORSTORE_DIR), help="向量索引输出目录")
    parser.add_argument("--with-chroma", action="store_true", help="额外构建 ChromaDB 索引")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_file = Path(args.chunks_file) if args.chunks_file else CHUNKS_DIR / f"all_{args.strategy}.json"
    if not chunks_file.exists():
        logger.error("找不到 %s，请先运行 chunk_documents.py", chunks_file)
        return

    with open(chunks_file, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("加载 %d 个 chunks（策略=%s）", len(chunks), args.strategy)

    client = get_client()
    build_faiss_index(chunks, client, output_dir)

    if args.with_chroma:
        build_chroma_index(chunks, client, output_dir)

    logger.info("\n索引构建完成！")
    logger.info("  FAISS 索引: %s", output_dir / "faiss_index.bin")
    logger.info("  元数据:     %s", output_dir / "faiss_meta.json")


if __name__ == "__main__":
    main()
