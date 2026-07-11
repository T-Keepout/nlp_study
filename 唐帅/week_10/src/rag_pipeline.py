"""
技术手册 RAG 问答流水线。

流程：
  (可选) 查询改写
        ↓
  向量检索（DashScope embedding + FAISS）
        +
  BM25 关键词检索（jieba + rank_bm25）
        ↓
  RRF 融合排名
        ↓
  (可选) CrossEncoder Rerank
        ↓
  相关性阈值过滤
        ↓
  LLM 生成（DashScope qwen-plus）+ 引用标注

使用方式：
  python rag_pipeline.py
  python rag_pipeline.py --query "如何启动 docker 容器"
  python rag_pipeline.py --query "如何启动 docker 容器" --source "CRPilot SDK.md"
  python rag_pipeline.py --query "..." --query-rewrite

依赖：
  pip install faiss-cpu rank_bm25 jieba openai numpy
  设置环境变量 DASHSCOPE_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "faiss_index.bin"
META_PATH = VECTORSTORE_DIR / "faiss_meta.json"

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024
LLM_MODEL = "qwen-plus"

TOP_K_RETRIEVE = 10
TOP_K_RERANK = 4
SCORE_THRESHOLD = 0.20

SYSTEM_PROMPT = """你是一个专业的技术文档问答助手，专门回答关于自动驾驶软件、SDK、工具链、部署和开发流程的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得编造资料外的信息
2. 如果参考资料不足，直接说“根据提供的资料无法回答此问题”
3. 如果涉及命令、参数、路径、版本，请尽量保持原文准确
4. 回答时优先给出结论，再补充必要步骤
5. 引用资料时在句末标注来源编号，如 [1]
6. 回答尽量简洁清楚，不要输出与问题无关的内容"""


def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


class VectorStore:
    def __init__(self, client: OpenAI):
        import faiss

        self.client = client
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)
        logger.info("FAISS 索引加载完成，共 %d 条向量", self.index.ntotal)

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=EMBED_MODEL,
            input=[query],
            dimensions=EMBED_DIM,
        )
        vec = np.array([resp.data[0].embedding], dtype="float32")
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        return vec

    def _match_filter(self, item: dict, filter_meta: dict) -> bool:
        for key, value in filter_meta.items():
            if key == "source_file":
                item_value = str(item.get(key, ""))
                normalized_target = str(value)
                if item_value != normalized_target and Path(item_value).stem != Path(normalized_target).stem:
                    return False
            elif str(item.get(key, "")) != str(value):
                return False
        return True

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        filter_meta: Optional[dict] = None,
    ) -> list[dict]:
        query_vec = self._embed_query(query)
        scores, indices = self.index.search(query_vec, top_k * 5)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_list):
                continue
            item = dict(self.meta_list[idx])
            item["vec_score"] = float(score)

            if filter_meta and not self._match_filter(item, filter_meta):
                continue

            results.append(item)
            if len(results) >= top_k:
                break
        return results


class BM25Store:
    def __init__(self):
        from rank_bm25 import BM25Okapi
        import jieba

        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        logger.info("构建 BM25 索引（分词中，请稍候）...")
        tokenized = [list(jieba.cut(item["content"])) for item in self.meta_list]
        self.bm25 = BM25Okapi(tokenized)
        self.jieba = jieba
        logger.info("BM25 索引完成")

    def _match_filter(self, item: dict, filter_meta: dict) -> bool:
        for key, value in filter_meta.items():
            if key == "source_file":
                item_value = str(item.get(key, ""))
                normalized_target = str(value)
                if item_value != normalized_target and Path(item_value).stem != Path(normalized_target).stem:
                    return False
            elif str(item.get(key, "")) != str(value):
                return False
        return True

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE, filter_meta: Optional[dict] = None) -> list[dict]:
        tokens = list(self.jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][: top_k * 5]

        results = []
        for idx in top_idx:
            if scores[idx] < 1e-9:
                continue
            item = dict(self.meta_list[idx])
            if filter_meta and not self._match_filter(item, filter_meta):
                continue
            item["bm25_score"] = float(scores[idx])
            results.append(item)
            if len(results) >= top_k:
                break
        return results


def reciprocal_rank_fusion(vec_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1 / (k + rank)
        chunk_map[cid] = item

    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1 / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = item
        else:
            chunk_map[cid].update(item)

    ranked = []
    for cid in sorted(rrf_scores, key=lambda key: -rrf_scores[key]):
        item = dict(chunk_map[cid])
        item["rrf_score"] = rrf_scores[cid]
        ranked.append(item)
    return ranked


def load_reranker_model():
    try:
        from sentence_transformers import CrossEncoder

        model_path = BASE_DIR / "models" / "bge-reranker-base"
        model_name = str(model_path) if model_path.exists() else "BAAI/bge-reranker-base"
        logger.info("加载 Reranker 模型: %s", model_name)
        reranker = CrossEncoder(model_name, device="cpu")
        logger.info("Reranker 模型加载完成")
        return reranker
    except ImportError:
        logger.warning("sentence-transformers 未安装，Rerank 已关闭")
    except Exception as exc:
        logger.warning("Reranker 模型加载失败，已自动关闭 Rerank: %s", exc)
    return None


def trim_for_rerank(text: str, max_chars: int = 1200) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def rerank(query: str, candidates: list[dict], reranker, top_k: int = TOP_K_RERANK) -> list[dict]:
    logger.info("开始 Rerank，候选数=%d", len(candidates))
    if reranker is None:
        logger.warning("Reranker 不可用，直接使用召回排序")
        return candidates[:top_k]

    try:
        pairs = [(query, trim_for_rerank(item["content"])) for item in candidates]
        scores = reranker.predict(
            pairs,
            batch_size=min(16, len(pairs)),
            show_progress_bar=False,
        )
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)
        candidates.sort(key=lambda item: -item.get("rerank_score", 0.0))
    except Exception as exc:
        logger.warning("Rerank 失败，使用原始排序: %s", exc)

    logger.info("Rerank 结束，保留 %d 条", min(top_k, len(candidates)))
    return candidates[:top_k]


def rewrite_query(query: str, client: OpenAI) -> str:
    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是技术文档检索优化专家。"
                    "请把用户问题改写成更适合从 SDK、部署文档、工具文档中检索的精确查询语句。"
                    "保留产品名、模块名、命令、版本、错误关键词，不要超过60字。"
                    "直接输出改写结果，不要解释。"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    rewritten = resp.choices[0].message.content.strip()
    logger.info("查询改写: %r -> %r", query, rewritten)
    return rewritten


def build_context(retrieved: list[dict]) -> tuple[str, list[dict]]:
    parts = []
    citations = []

    for i, item in enumerate(retrieved, 1):
        label = f"[{i}] {item.get('source_file', '')}"
        if item.get("section"):
            label += f" · {item['section']}"
        line_start = item.get("line_start", 0)
        line_end = item.get("line_end", 0)
        if line_start and line_end:
            label += f" · 行{line_start}-{line_end}"

        content = item.get("parent_content") or item.get("content", "")
        parts.append(f"{label}\n{content}")
        citations.append(
            {
                "index": i,
                "source": label,
                "chunk_id": item.get("chunk_id", ""),
            }
        )

    return "\n\n---\n\n".join(parts), citations


def call_llm(query: str, context: str, client: OpenAI) -> str:
    logger.info("开始调用 LLM 生成答案...")
    user_message = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        "请根据参考资料回答，并在引用内容处标注来源编号，例如 [1]。"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    logger.info("LLM 生成完成")
    return resp.choices[0].message.content


class RAGPipeline:
    def __init__(
        self,
        *,
        use_bm25: bool = True,
        use_rerank: bool = True,
        use_query_rewrite: bool = False,
    ):
        self.client = get_client()
        self.vec_store = VectorStore(self.client)
        self.use_bm25 = use_bm25
        self.use_rerank = use_rerank
        self.use_qr = use_query_rewrite
        self.bm25_store = None
        self.reranker = None

        if use_bm25:
            try:
                self.bm25_store = BM25Store()
            except ImportError:
                logger.warning("缺少 jieba 或 rank_bm25，BM25 已自动关闭")
                self.use_bm25 = False

        if use_rerank:
            self.reranker = load_reranker_model()
            if self.reranker is None:
                self.use_rerank = False

    def query(
        self,
        question: str,
        *,
        filter_meta: Optional[dict] = None,
        verbose: bool = False,
    ) -> dict:
        retrieval_query = rewrite_query(question, self.client) if self.use_qr else question

        vec_results = self.vec_store.search(retrieval_query, TOP_K_RETRIEVE, filter_meta)
        if verbose:
            if vec_results:
                logger.info("向量召回: %d 条，最高分=%.3f", len(vec_results), vec_results[0].get("vec_score", 0.0))
            else:
                logger.info("向量召回: 0 条")

        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(retrieval_query, TOP_K_RETRIEVE, filter_meta)
            candidates = reciprocal_rank_fusion(vec_results, bm25_results)
            if verbose:
                logger.info("BM25 召回: %d 条，RRF 后: %d 条", len(bm25_results), len(candidates))
        else:
            candidates = vec_results

        if verbose:
            logger.info("进入后处理阶段：%s%s",
                        "Rerank " if self.use_rerank else "",
                        "LLM 生成")
        final = rerank(question, candidates, self.reranker, TOP_K_RERANK) if self.use_rerank else candidates[:TOP_K_RERANK]

        if verbose:
            logger.info("最终使用 %d 条上下文", len(final))

        if not final:
            return {
                "answer": "未找到相关内容，无法回答此问题。",
                "citations": [],
                "retrieved": [],
            }

        top_score = final[0].get("vec_score", final[0].get("rerank_score", 1.0))
        if top_score < SCORE_THRESHOLD and filter_meta is None:
            return {
                "answer": "根据当前知识库未能找到与该问题相关的内容，建议换一个更具体的问题，或指定文档来源。",
                "citations": [],
                "retrieved": final,
            }

        context, citations = build_context(final)
        answer = call_llm(question, context, self.client)
        return {
            "answer": answer,
            "citations": citations,
            "retrieved": final,
        }


def print_result(question: str, result: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"问题：{question}")
    print(f"{'=' * 60}\n")
    print(result["answer"])
    if result["citations"]:
        print("\n-- 来源 --")
        for citation in result["citations"]:
            print(f"  {citation['source']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="技术手册 RAG 问答")
    parser.add_argument("--query", type=str, default=None, help="用户问题")
    parser.add_argument("--source", type=str, default=None, help="限定来源文档，如 CRPilot SDK.md")
    parser.add_argument("--query-rewrite", action="store_true", help="开启查询改写")
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25")
    parser.add_argument("--no-rerank", action="store_true", help="关闭 Rerank")
    args = parser.parse_args()

    pipeline = RAGPipeline(
        use_bm25=not args.no_bm25,
        use_rerank=not args.no_rerank,
        use_query_rewrite=args.query_rewrite,
    )

    filter_meta = None
    if args.source:
        filter_meta = {"source_file": args.source}

    if args.query:
        result = pipeline.query(args.query, filter_meta=filter_meta, verbose=True)
        print_result(args.query, result)
        return

    print("技术手册 RAG 问答系统")
    print(f"模型：{LLM_MODEL}  |  向量库：{INDEX_PATH}")
    print("输入 'exit' 退出，输入 'mode' 查看当前配置\n")

    while True:
        try:
            question = input("问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() == "exit":
            break
        if question.lower() == "mode":
            print(
                f"BM25={'on' if pipeline.use_bm25 else 'off'}  "
                f"Rerank={'on' if pipeline.use_rerank else 'off'}  "
                f"QueryRewrite={'on' if pipeline.use_qr else 'off'}"
            )
            continue

        result = pipeline.query(question, filter_meta=filter_meta, verbose=True)
        print_result(question, result)


if __name__ == "__main__":
    main()
