"""
技术手册向量索引构建脚本（LangChain 版）

与原生版 src/build_index.py 的区别：
  - 原生版：手动 embedding + 手动 FAISS + 手动 metadata 管理
  - LangChain 版：把现成 chunk 封装成 Document，由 LangChain 接管向量库构建

输入：
  data/chunks/all_semantic.json

输出：
  vectorstore/faiss_lc/

依赖：
  pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers
"""

from __future__ import annotations

import json
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_lc"
MODELS_DIR = BASE_DIR / "models"
BGE_MODEL_PATH = MODELS_DIR / "bge-small-zh-v1.5"
CHUNKS_FILE = CHUNKS_DIR / "all_semantic.json"


def load_documents():
    """
    从现成的 chunk JSON 构造 LangChain Document。

    这样可以直接复用你前面已经做好的：
      parse_manuals.py -> chunk_documents.py
    """
    from langchain_core.documents import Document

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"找不到 {CHUNKS_FILE}，请先运行 src/chunk_documents.py")

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    docs = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata", {}))
        metadata["chunk_id"] = chunk.get("chunk_id", "")
        docs.append(
            Document(
                page_content=chunk.get("content", ""),
                metadata=metadata,
            )
        )

    logger.info("加载完成：%d 个 chunk -> %d 个 Document", len(chunks), len(docs))
    return docs


def get_embeddings():
    """
    使用本地 bge-small-zh-v1.5 作为 embedding 模型。
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    model_path = str(BGE_MODEL_PATH) if BGE_MODEL_PATH.exists() else "BAAI/bge-small-zh-v1.5"
    if not BGE_MODEL_PATH.exists():
        logger.warning("本地模型不存在: %s，将尝试从 HuggingFace 下载", BGE_MODEL_PATH)

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=str(MODELS_DIR),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding 模型加载完成: %s", model_path)
    return embeddings


def build_vectorstore(docs, embeddings):
    from langchain_community.vectorstores import FAISS

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("构建 LangChain FAISS 向量库（%d 个 Document）...", len(docs))
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    logger.info("向量库已保存 -> %s", VECTORSTORE_DIR)
    logger.info("  index.faiss: %d KB", (VECTORSTORE_DIR / "index.faiss").stat().st_size // 1024)
    return vectorstore


def main():
    docs = load_documents()
    embeddings = get_embeddings()
    build_vectorstore(docs, embeddings)

    print("\nLangChain 技术手册向量库构建完成！")
    print(f"  路径: {VECTORSTORE_DIR}")
    print(f"  下一步: python src_langhcain/rag_chain_lc.py")


if __name__ == "__main__":
    main()
