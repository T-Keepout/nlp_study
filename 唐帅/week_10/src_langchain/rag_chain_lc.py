"""
技术手册 RAG 问答链（LangChain LCEL 版）

特点：
  - 使用本地 BGE embedding + LangChain FAISS 向量库
  - 支持按 source_file 过滤文档
  - 支持返回来源文档片段
  - 结构尽量保持和课件版一致

使用方式：
  python rag_chain_lc.py
  python rag_chain_lc.py --query "如何启动 docker 容器"
  python rag_chain_lc.py --query "如何启动 docker 容器" --source "CRPilot SDK.md"
  python rag_chain_lc.py --query "..." --with-sources

依赖：
  pip install langchain langchain-openai langchain-community langchain-huggingface faiss-cpu
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_lc"
MODELS_DIR = BASE_DIR / "models"
BGE_MODEL_PATH = MODELS_DIR / "bge-small-zh-v1.5"

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"

SYSTEM_PROMPT = """你是一个专业的技术文档问答助手，专门回答关于自动驾驶软件、SDK、工具链、部署和开发流程的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得编造资料外的信息
2. 如果参考资料不足，直接说“根据提供的资料无法回答此问题”
3. 如果涉及命令、参数、路径、版本，请尽量保持原文准确
4. 回答时优先给出结论，再补充必要步骤
5. 引用资料时在句末标注来源编号，如 [1]
6. 回答尽量简洁清楚，不要输出与问题无关的内容"""


def get_llm():
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")

    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=api_key,
        openai_api_base=DASHSCOPE_URL,
        temperature=0.1,
    )


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    model_path = str(BGE_MODEL_PATH) if BGE_MODEL_PATH.exists() else "BAAI/bge-small-zh-v1.5"
    return HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=str(MODELS_DIR),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(embeddings):
    from langchain_community.vectorstores import FAISS

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"向量库不存在: {VECTORSTORE_DIR}\n请先运行: python src_langhcain/build_index_lc.py"
        )

    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_retriever(vectorstore, source_file: str | None = None, top_k: int = 4):
    """
    自定义 retriever：
      - 无 source 时直接 similarity_search
      - 有 source 时先多召回，再按 metadata.source_file 过滤
    """
    def retrieve(question: str):
        fetch_k = top_k * 6 if source_file else top_k
        docs = vectorstore.similarity_search(question, k=fetch_k)
        if source_file:
            filtered = [
                doc for doc in docs
                if str(doc.metadata.get("source_file", "")) == source_file
                or Path(str(doc.metadata.get("source_file", ""))).stem == Path(source_file).stem
            ]
            docs = filtered[:top_k]
        else:
            docs = docs[:top_k]
        return docs

    return retrieve


def format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        label = f"[{i}] {meta.get('source_file', '')}"
        if meta.get("section"):
            label += f" · {meta['section']}"
        line_start = meta.get("line_start", 0)
        line_end = meta.get("line_end", 0)
        if line_start and line_end:
            label += f" · 行{line_start}-{line_end}"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain(vectorstore, source_file: str | None = None):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    llm = get_llm()
    retrieve = build_retriever(vectorstore, source_file=source_file, top_k=4)
    retriever = RunnableLambda(retrieve)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}\n\n请根据参考资料回答，标注来源编号。"),
    ])

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def build_chain_with_sources(vectorstore, source_file: str | None = None):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

    llm = get_llm()
    retrieve = build_retriever(vectorstore, source_file=source_file, top_k=4)
    retriever = RunnableLambda(retrieve)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}"),
    ])

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_sources = RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(answer=rag_chain_from_docs)

    return chain_with_sources


def print_sources(docs):
    print("\n-- 来源文档片段 --")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        label = f"[{i}] {meta.get('source_file', '')}"
        if meta.get("section"):
            label += f" · {meta['section']}"
        line_start = meta.get("line_start", 0)
        line_end = meta.get("line_end", 0)
        if line_start and line_end:
            label += f" · 行{line_start}-{line_end}"
        print(label)
        print(f"    {doc.page_content[:160].replace(chr(10), ' ')}...")


def main():
    parser = argparse.ArgumentParser(description="技术手册 RAG 问答（LangChain LCEL 版）")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--source", type=str, default=None, help="限定来源文档，如 CRPilot SDK.md")
    parser.add_argument("--with-sources", action="store_true", help="输出来源文档片段")
    args = parser.parse_args()

    logger.info("加载 embedding 模型...")
    embeddings = get_embeddings()
    logger.info("加载向量库...")
    vectorstore = get_vectorstore(embeddings)

    if args.with_sources:
        chain = build_chain_with_sources(vectorstore, source_file=args.source)
    else:
        chain, _ = build_chain(vectorstore, source_file=args.source)

    def run_query(question: str):
        print(f"\n{'=' * 60}")
        print(f"问题：{question}")
        print(f"{'=' * 60}")

        if args.with_sources:
            result = chain.invoke(question)
            print(f"\n{result['answer']}")
            print_sources(result["context"])
        else:
            answer = chain.invoke(question)
            print(f"\n{answer}")

    if args.query:
        run_query(args.query)
    else:
        print("技术手册 RAG 问答系统（LangChain LCEL 版）")
        print(f"模型：{LLM_MODEL}  |  向量库：{VECTORSTORE_DIR}")
        print("输入 'exit' 退出\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() == "exit":
                break
            run_query(q)


if __name__ == "__main__":
    main()
