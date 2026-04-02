from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import chromadb
import httpx
import requests
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
DEFAULT_KB_DIR = ROOT / "knowledge_base" / "tsinghua_stats"
DEFAULT_DB_DIR = ROOT / "rag_store" / "stats_chroma"
DEFAULT_MANIFEST_PATH = ROOT / "rag_store" / "stats_manifest.json"
COLLECTION_NAME = "tsinghua_stats_v1"


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean, got {value!r}")


@dataclass
class AppConfig:
    generation_api_key: str
    generation_base_url: str
    generation_model: str
    generation_timeout: int
    generation_enable_thinking: bool
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    embedding_dimensions: int | None
    rerank_api_key: str
    rerank_base_url: str
    rerank_model: str
    retrieval_top_k: int
    rerank_top_n: int
    chunk_target_chars: int
    chunk_overlap_pages: int
    min_chunk_chars: int
    embedding_query_instruction: str
    rerank_instruction: str
    knowledge_base_dir: Path
    chroma_dir: Path
    manifest_path: Path

    @classmethod
    def load(cls) -> "AppConfig":
        load_dotenv(ROOT / ".env")

        generation_api_key = os.getenv("GENERATION_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        generation_base_url = os.getenv("GENERATION_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
        generation_model = os.getenv("GENERATION_MODEL") or os.getenv("OPENAI_MODEL") or ""

        embedding_api_key = os.getenv("EMBEDDING_API_KEY") or ""
        embedding_base_url = os.getenv("EMBEDDING_BASE_URL") or ""
        embedding_model = os.getenv("EMBEDDING_MODEL") or ""

        rerank_api_key = os.getenv("RERANK_API_KEY") or ""
        rerank_base_url = os.getenv("RERANK_BASE_URL") or ""
        rerank_model = os.getenv("RERANK_MODEL") or ""

        required = {
            "GENERATION_API_KEY or OPENAI_API_KEY": generation_api_key,
            "GENERATION_BASE_URL or OPENAI_BASE_URL": generation_base_url,
            "GENERATION_MODEL or OPENAI_MODEL": generation_model,
            "EMBEDDING_API_KEY": embedding_api_key,
            "EMBEDDING_BASE_URL": embedding_base_url,
            "EMBEDDING_MODEL": embedding_model,
            "RERANK_API_KEY": rerank_api_key,
            "RERANK_BASE_URL": rerank_base_url,
            "RERANK_MODEL": rerank_model,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing required environment variables: {joined}")

        embedding_dimensions_raw = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
        embedding_dimensions = int(embedding_dimensions_raw) if embedding_dimensions_raw else 1024

        return cls(
            generation_api_key=generation_api_key,
            generation_base_url=generation_base_url,
            generation_model=generation_model,
            generation_timeout=env_int("OPENAI_TIMEOUT", 60),
            generation_enable_thinking=env_bool("GENERATION_ENABLE_THINKING", False),
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            rerank_api_key=rerank_api_key,
            rerank_base_url=rerank_base_url,
            rerank_model=rerank_model,
            retrieval_top_k=env_int("RETRIEVAL_TOP_K", 12),
            rerank_top_n=env_int("RERANK_TOP_N", 5),
            chunk_target_chars=env_int("CHUNK_TARGET_CHARS", 550),
            chunk_overlap_pages=env_int("CHUNK_OVERLAP_PAGES", 1),
            min_chunk_chars=env_int("MIN_CHUNK_CHARS", 120),
            embedding_query_instruction=os.getenv(
                "EMBEDDING_QUERY_INSTRUCTION",
                "Retrieve passages from a statistics course knowledge base that best answer the user's question.",
            ),
            rerank_instruction=os.getenv(
                "RERANK_INSTRUCTION",
                "Please rerank the documents based on how well they answer the query about statistics concepts, methods, assumptions, and interpretation.",
            ),
            knowledge_base_dir=Path(os.getenv("KNOWLEDGE_BASE_DIR", str(DEFAULT_KB_DIR))),
            chroma_dir=Path(os.getenv("CHROMA_DB_DIR", str(DEFAULT_DB_DIR))),
            manifest_path=Path(os.getenv("RAG_MANIFEST_PATH", str(DEFAULT_MANIFEST_PATH))),
        )


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u3000", " ")
    text = text.replace("\ufeff", " ")
    text = text.replace("•", "• ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def is_probable_boilerplate(line: str, count: int, total_pages: int) -> bool:
    if not line:
        return True
    if re.fullmatch(r"\d{1,3}", line):
        return True
    if len(line) <= 2 and count >= 3:
        return True
    threshold = max(5, int(total_pages * 0.35))
    if count >= threshold and len(line) <= 40:
        return True
    return False


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    page_lines: list[list[str]] = []
    frequency: dict[str, int] = {}

    for page in reader.pages:
        raw_text = page.extract_text() or ""
        lines = [normalize_line(line) for line in raw_text.splitlines()]
        lines = [line for line in lines if line]
        page_lines.append(lines)
        for line in set(lines):
            frequency[line] = frequency.get(line, 0) + 1

    pages: list[dict[str, Any]] = []
    total_pages = len(page_lines)
    for index, lines in enumerate(page_lines, start=1):
        filtered = [
            line
            for line in lines
            if not is_probable_boilerplate(line, frequency.get(line, 0), total_pages)
        ]
        text = clean_text("\n".join(filtered))
        if not text:
            continue
        pages.append(
            {
                "page_number": index,
                "text": text,
                "char_count": len(text),
            }
        )
    return pages


def chunk_pages(
    pages: Sequence[dict[str, Any]],
    pdf_name: str,
    target_chars: int,
    overlap_pages: int,
    min_chunk_chars: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if not pages:
        return chunks

    start = 0
    chunk_id = 0
    while start < len(pages):
        end = start
        char_total = 0
        chunk_pages_buffer: list[dict[str, Any]] = []

        while end < len(pages):
            candidate = pages[end]
            candidate_text = candidate["text"]
            projected = char_total + len(candidate_text) + (2 if chunk_pages_buffer else 0)
            if chunk_pages_buffer and projected > target_chars:
                break
            chunk_pages_buffer.append(candidate)
            char_total = projected
            end += 1

        if not chunk_pages_buffer:
            chunk_pages_buffer.append(pages[start])
            end = start + 1

        text = "\n\n".join(page["text"] for page in chunk_pages_buffer)
        if len(text) >= min_chunk_chars or not chunks:
            page_numbers = [page["page_number"] for page in chunk_pages_buffer]
            chunks.append(
                {
                    "id": f"{pdf_name}:{chunk_id:04d}",
                    "text": clean_text(text),
                    "metadata": {
                        "source": pdf_name,
                        "page_start": min(page_numbers),
                        "page_end": max(page_numbers),
                        "page_span": len(page_numbers),
                    },
                }
            )
            chunk_id += 1

        if end >= len(pages):
            break
        start = max(start + 1, end - overlap_pages)

    return chunks


def iter_source_pdfs(knowledge_base_dir: Path) -> list[Path]:
    return sorted(path for path in knowledge_base_dir.glob("*.pdf") if path.is_file())


def build_chunks(config: AppConfig) -> list[dict[str, Any]]:
    pdf_paths = iter_source_pdfs(config.knowledge_base_dir)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {config.knowledge_base_dir}")

    all_chunks: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        pages = extract_pdf_pages(pdf_path)
        chunks = chunk_pages(
            pages=pages,
            pdf_name=pdf_path.name,
            target_chars=config.chunk_target_chars,
            overlap_pages=config.chunk_overlap_pages,
            min_chunk_chars=config.min_chunk_chars,
        )
        all_chunks.extend(chunks)
    return all_chunks


class SiliconFlowEmbedder:
    def __init__(self, config: AppConfig) -> None:
        self.http_client = httpx.Client(trust_env=False, timeout=120.0)
        self.client = OpenAI(
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url,
            http_client=self.http_client,
        )
        self.model = config.embedding_model
        self.dimensions = config.embedding_dimensions
        self.query_instruction = config.embedding_query_instruction

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            embeddings.extend(self._embed(texts[start : start + batch_size]))
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        prompt = f"Instruct: {self.query_instruction}\nQuery: {query}"
        return self._embed([prompt])[0]


class SiliconFlowReranker:
    def __init__(self, config: AppConfig) -> None:
        self.url = config.rerank_base_url.rstrip("/") + "/rerank"
        self.model = config.rerank_model
        self.api_key = config.rerank_api_key
        self.instruction = config.rerank_instruction
        self.session = requests.Session()
        self.session.trust_env = False

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[dict[str, Any]]:
        payload = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n,
            "return_documents": True,
            "instruction": self.instruction,
        }
        response = self.session.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])


class Generator:
    def __init__(self, config: AppConfig) -> None:
        self.http_client = httpx.Client(trust_env=False, timeout=float(config.generation_timeout))
        self.client = OpenAI(
            api_key=config.generation_api_key,
            base_url=config.generation_base_url,
            http_client=self.http_client,
        )
        self.model = config.generation_model
        self.timeout = config.generation_timeout
        self.enable_thinking = config.generation_enable_thinking

    def answer(self, question: str, contexts: Sequence[dict[str, Any]]) -> str:
        context_blocks = []
        for idx, item in enumerate(contexts, start=1):
            meta = item["metadata"]
            label = f"[{idx}] {meta['source']} 第 {meta['page_start']}-{meta['page_end']} 页"
            context_blocks.append(f"{label}\n{item['text']}")
        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "你是一个应用统计课程助教。"
            "请严格依据检索到的课程讲义内容回答，优先给出准确、清晰、适合课堂展示的中文解释。"
            "如果证据不足，请明确说“根据当前知识库无法完全确认”。"
            "回答时尽量说明概念、适用场景、关键假设和结论解释。"
            "如果需要写公式，请优先使用 LaTeX 语法，用 $...$ 或 $$...$$ 包裹。"
            "在结尾附上你引用的资料编号。"
        )
        user_prompt = (
            f"问题：{question}\n\n"
            f"可用资料：\n\n{context_text}\n\n"
            "请给出结构清晰的回答。"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
            extra_body={"enable_thinking": self.enable_thinking},
            timeout=self.timeout,
        )
        return response.choices[0].message.content or ""


def ensure_dirs(config: AppConfig) -> None:
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)


def get_collection(config: AppConfig) -> Collection:
    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def rebuild_index(config: AppConfig, force: bool = False) -> dict[str, Any]:
    ensure_dirs(config)
    collection = get_collection(config)
    if force:
        existing = collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)

    chunks = build_chunks(config)
    embedder = SiliconFlowEmbedder(config)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_documents(texts)
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    existing = collection.get(include=[])
    existing_ids = set(existing.get("ids", []))
    new_records = [
        (record_id, document, metadata, embedding)
        for record_id, document, metadata, embedding in zip(ids, texts, metadatas, embeddings)
        if force or record_id not in existing_ids
    ]
    if new_records:
        collection.add(
            ids=[item[0] for item in new_records],
            documents=[item[1] for item in new_records],
            metadatas=[item[2] for item in new_records],
            embeddings=[item[3] for item in new_records],
        )

    manifest = {
        "collection_name": COLLECTION_NAME,
        "knowledge_base_dir": str(config.knowledge_base_dir),
        "pdf_count": len(iter_source_pdfs(config.knowledge_base_dir)),
        "chunk_count": len(chunks),
        "embedding_model": config.embedding_model,
        "embedding_dimensions": config.embedding_dimensions,
        "rerank_model": config.rerank_model,
        "generation_model": config.generation_model,
        "chunk_target_chars": config.chunk_target_chars,
        "chunk_overlap_pages": config.chunk_overlap_pages,
        "min_chunk_chars": config.min_chunk_chars,
    }
    config.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def search(config: AppConfig, query: str) -> list[dict[str, Any]]:
    collection = get_collection(config)
    count = collection.count()
    if count == 0:
        raise RuntimeError("向量库为空，请先运行 build 命令。")

    embedder = SiliconFlowEmbedder(config)
    reranker = SiliconFlowReranker(config)

    query_embedding = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(config.retrieval_top_k, max(count, 1)),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = [
        {"text": text, "metadata": metadata, "distance": distance}
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]
    reranked = reranker.rerank(
        query=query,
        documents=[item["text"] for item in retrieved],
        top_n=min(config.rerank_top_n, len(retrieved)),
    )

    final_items: list[dict[str, Any]] = []
    for item in reranked:
        index = item["index"]
        source = retrieved[index]
        final_items.append(
            {
                "text": source["text"],
                "metadata": source["metadata"],
                "distance": source["distance"],
                "relevance_score": item.get("relevance_score"),
            }
        )
    return final_items


def format_sources(contexts: Sequence[dict[str, Any]]) -> str:
    lines = []
    for idx, item in enumerate(contexts, start=1):
        meta = item["metadata"]
        lines.append(
            f"[{idx}] {meta['source']} 第 {meta['page_start']}-{meta['page_end']} 页 "
            f"(rerank={item.get('relevance_score', 0):.4f})"
        )
    return "\n".join(lines)


def cmd_build(args: argparse.Namespace, config: AppConfig) -> int:
    manifest = rebuild_index(config=config, force=args.force)
    print("知识库构建完成。")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace, config: AppConfig) -> int:
    contexts = search(config, args.question)
    generator = Generator(config)
    answer = generator.answer(args.question, contexts)
    print("\n=== 回答 ===\n")
    print(answer.strip())
    print("\n=== 引用来源 ===\n")
    print(format_sources(contexts))
    return 0


def cmd_search(args: argparse.Namespace, config: AppConfig) -> int:
    contexts = search(config, args.question)
    print(format_sources(contexts))
    print("\n=== 检索片段 ===\n")
    for idx, item in enumerate(contexts, start=1):
        snippet = textwrap.shorten(item["text"].replace("\n", " "), width=260, placeholder="...")
        print(f"[{idx}] {snippet}")
    return 0


def cmd_shell(_: argparse.Namespace, config: AppConfig) -> int:
    generator = Generator(config)
    print("统计学 RAG 交互模式。输入 exit 或 quit 退出。")
    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("退出。")
            return 0
        contexts = search(config, question)
        answer = generator.answer(question, contexts)
        print("\n回答：\n")
        print(answer.strip())
        print("\n来源：")
        print(format_sources(contexts))


def cmd_check(_: argparse.Namespace, config: AppConfig) -> int:
    print("配置检查通过。")
    print(f"知识库目录: {config.knowledge_base_dir}")
    print(f"向量库存储: {config.chroma_dir}")
    print(f"Embedding 模型: {config.embedding_model}")
    print(f"Rerank 模型: {config.rerank_model}")
    print(f"回答模型: {config.generation_model}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="应用统计中文课程 RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_check = subparsers.add_parser("check", help="检查配置")
    parser_check.set_defaults(func=cmd_check)

    parser_build = subparsers.add_parser("build", help="构建知识库索引")
    parser_build.add_argument("--force", action="store_true", help="重建索引")
    parser_build.set_defaults(func=cmd_build)

    parser_search = subparsers.add_parser("search", help="只做检索与 rerank")
    parser_search.add_argument("question", help="检索问题")
    parser_search.set_defaults(func=cmd_search)

    parser_ask = subparsers.add_parser("ask", help="检索并生成回答")
    parser_ask.add_argument("question", help="提问内容")
    parser_ask.set_defaults(func=cmd_ask)

    parser_shell = subparsers.add_parser("shell", help="进入交互式问答")
    parser_shell.set_defaults(func=cmd_shell)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = AppConfig.load()
        return args.func(args, config)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
