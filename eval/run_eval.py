from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stats_rag import (  # noqa: E402
    AppConfig,
    Generator,
    SiliconFlowEmbedder,
    SiliconFlowReranker,
    env_bool,
    env_int,
    get_collection,
)


DEFAULT_DATASET_PATH = ROOT / "eval" / "eval_set_template.jsonl"
DEFAULT_RUNS_DIR = ROOT / "eval" / "runs"
DEFAULT_RETRIEVAL_TOP_K = 12
DEFAULT_RERANK_TOP_N = 5


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {value!r}") from exc


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def average(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def coerce_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("Judge model returned empty content.")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Judge model response is not a JSON object.")
    return parsed


def round_metric(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


@dataclass
class EvalSample:
    id: str
    question: str
    question_type: str
    difficulty: str
    should_refuse: bool
    gold_evidence: list[dict[str, Any]]
    reference_answer: str
    key_points: list[str]
    notes: str
    tags: list[str]


@dataclass
class JudgeConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    enable_thinking: bool
    max_tokens: int
    response_format: str
    timeout: int

    @classmethod
    def load(cls) -> "JudgeConfig":
        load_dotenv(ROOT / ".env")

        api_key = (
            os.getenv("JUDGE_API_KEY")
            or os.getenv("GENERATION_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        base_url = (
            os.getenv("JUDGE_BASE_URL")
            or os.getenv("GENERATION_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or ""
        )
        model = os.getenv("JUDGE_MODEL") or ""
        required = {
            "JUDGE_API_KEY or GENERATION_API_KEY or OPENAI_API_KEY": api_key,
            "JUDGE_BASE_URL or GENERATION_BASE_URL or OPENAI_BASE_URL": base_url,
            "JUDGE_MODEL": model,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing required judge environment variables: {joined}")

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=env_float("JUDGE_TEMPERATURE", 0.0),
            enable_thinking=env_bool("JUDGE_ENABLE_THINKING", False),
            max_tokens=env_int("JUDGE_MAX_TOKENS", 600),
            response_format=os.getenv("JUDGE_RESPONSE_FORMAT", "json_object").strip() or "json_object",
            timeout=env_int("JUDGE_TIMEOUT", 120),
        )


class RAGRunner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.collection = get_collection(config)
        self.embedder = SiliconFlowEmbedder(config)
        self.reranker = SiliconFlowReranker(config)
        self.generator = Generator(config)

    def ensure_ready(self) -> None:
        count = self.collection.count()
        if count == 0:
            raise RuntimeError("Vector store is empty. Please run `python stats_rag.py build` first.")

    def search(self, query: str) -> list[dict[str, Any]]:
        count = self.collection.count()
        query_embedding = self.embedder.embed_query(query)
        retrieved = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(self.config.retrieval_top_k, max(count, 1)),
            include=["documents", "metadatas", "distances"],
        )

        documents = retrieved["documents"][0]
        metadatas = retrieved["metadatas"][0]
        distances = retrieved["distances"][0]
        candidates = [
            {"text": text, "metadata": metadata, "distance": distance}
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]
        reranked = self.reranker.rerank(
            query=query,
            documents=[item["text"] for item in candidates],
            top_n=min(self.config.rerank_top_n, len(candidates)),
        )

        final_items: list[dict[str, Any]] = []
        for item in reranked:
            index = item["index"]
            source = candidates[index]
            final_items.append(
                {
                    "text": source["text"],
                    "metadata": source["metadata"],
                    "distance": source["distance"],
                    "relevance_score": item.get("relevance_score"),
                }
            )
        return final_items

    def answer(self, question: str, contexts: Sequence[dict[str, Any]]) -> str:
        return self.generator.answer(question, contexts).strip()


class JudgeClient:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self.http_client = httpx.Client(trust_env=False, timeout=float(config.timeout))
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            http_client=self.http_client,
        )

    def evaluate(
        self,
        sample: EvalSample,
        contexts: Sequence[dict[str, Any]],
        answer: str,
    ) -> dict[str, Any]:
        key_points_text = (
            "\n".join(f"- {item}" for item in sample.key_points) if sample.key_points else "- 无"
        )
        evidence_text = build_context_for_judge(contexts)
        reference_answer = sample.reference_answer or "无"

        system_prompt = (
            "你是一个严谨的中文统计学 RAG 评测裁判。"
            "你只能依据提供的问题、标准答案、关键点、检索证据和模型回答进行评分，"
            "不要用外部常识替代证据，不要输出任何 JSON 以外的文本。"
        )

        user_prompt = f"""
请评估下面这条 RAG 问答样本，并只输出一个 JSON 对象。

判定目标：
1. `predicted_decision` 只能是 `\"answer\"` 或 `\"refuse\"`。
2. 如果模型回答主体是在说“知识库无法确认、没有相关内容、无法回答”，则判为 `\"refuse\"`。
3. 对于 `should_refuse=false` 的样本：
   - 如果 `predicted_decision=\"refuse\"`，则 `answer_accuracy_score` 必须为 0。
   - `key_point_results` 必须逐条判断每个关键点是否被回答覆盖。
   - `faithfulness` 只能是 0 或 1。
   - 如果只是保守拒答、没有编造事实，`faithfulness` 可以为 1；回答正确性由 `answer_accuracy_score` 体现。
4. 对于 `should_refuse=true` 的样本：
   - `answer_accuracy_score` 设为 null。
   - `key_point_results` 设为空数组。
   - `faithfulness` 设为 null。
   - `refusal_appropriate` 只能是 0 或 1。
5. `notes` 请控制在 80 个中文字符以内，简要说明主要原因。

评分标准：
- `answer_accuracy_score` 取值 0 到 4：
  - 4：正确、完整、无明显错误
  - 3：基本正确，但有少量次要遗漏
  - 2：部分正确，但遗漏关键结论或有明显不严谨
  - 1：只有少量相关内容，整体没有答到核心
  - 0：错误、幻觉明显，或没有回答问题
- `faithfulness=1` 表示回答中的关键结论能被给定检索证据支持，且没有重要编造。
- `faithfulness=0` 表示有重要结论无法由检索证据支持，或与证据冲突。

请严格输出如下结构：
{{
  "predicted_decision": "answer",
  "answer_accuracy_score": 0,
  "key_point_results": [
    {{"key_point": "示例", "covered": true}}
  ],
  "faithfulness": 1,
  "refusal_appropriate": null,
  "notes": "简短说明"
}}

样本信息：
- sample_id: {sample.id}
- question_type: {sample.question_type}
- difficulty: {sample.difficulty}
- should_refuse: {str(sample.should_refuse).lower()}

问题：
{sample.question}

标准答案：
{reference_answer}

关键点：
{key_points_text}

检索证据：
{evidence_text}

模型回答：
{answer or "(空回答)"}
""".strip()

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "extra_body": {"enable_thinking": self.config.enable_thinking},
            "timeout": self.config.timeout,
        }
        if self.config.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        raw_content = coerce_text(response.choices[0].message.content)
        parsed = extract_json_object(raw_content)
        return normalize_judge_result(sample, parsed, raw_content)


def build_context_for_judge(contexts: Sequence[dict[str, Any]]) -> str:
    if not contexts:
        return "(无检索证据)"
    blocks: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        meta = item["metadata"]
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] source={meta['source']} pages={meta['page_start']}-{meta['page_end']} "
                    f"rerank={item.get('relevance_score', 0):.4f}",
                    item["text"],
                ]
            )
        )
    return "\n\n".join(blocks)


def normalize_judge_result(sample: EvalSample, parsed: dict[str, Any], raw_content: str) -> dict[str, Any]:
    predicted_decision = str(parsed.get("predicted_decision", "answer")).strip().lower()
    if predicted_decision not in {"answer", "refuse"}:
        predicted_decision = "answer"

    score = parsed.get("answer_accuracy_score")
    if sample.should_refuse:
        normalized_score = None
    else:
        try:
            normalized_score = max(0, min(4, int(score)))
        except (TypeError, ValueError):
            normalized_score = 0
        if predicted_decision == "refuse":
            normalized_score = 0

    raw_key_point_results = parsed.get("key_point_results")
    if sample.should_refuse:
        key_point_results: list[dict[str, Any]] = []
    else:
        key_point_results = []
        if isinstance(raw_key_point_results, list):
            for index, key_point in enumerate(sample.key_points):
                item = raw_key_point_results[index] if index < len(raw_key_point_results) else {}
                covered = False
                if isinstance(item, dict):
                    covered = bool(item.get("covered", False))
                    judged_point = str(item.get("key_point", "")).strip() or key_point
                else:
                    judged_point = key_point
                key_point_results.append({"key_point": judged_point, "covered": covered})
        else:
            key_point_results = [{"key_point": key_point, "covered": False} for key_point in sample.key_points]

    faithfulness = parsed.get("faithfulness")
    if sample.should_refuse:
        normalized_faithfulness = None
    else:
        normalized_faithfulness = 1 if str(faithfulness).strip() in {"1", "true", "True"} else 0

    refusal_appropriate = parsed.get("refusal_appropriate")
    if sample.should_refuse:
        normalized_refusal_appropriate = (
            1 if str(refusal_appropriate).strip() in {"1", "true", "True"} else 0
        )
    else:
        normalized_refusal_appropriate = None

    notes = str(parsed.get("notes", "")).strip()
    if not notes:
        notes = "Judge 未给出说明。"

    return {
        "predicted_decision": predicted_decision,
        "answer_accuracy_score": normalized_score,
        "answer_accuracy": safe_div(float(normalized_score), 4.0) if normalized_score is not None else None,
        "key_point_results": key_point_results,
        "key_point_coverage": safe_div(
            sum(1 for item in key_point_results if item["covered"]),
            len(key_point_results),
        )
        if key_point_results
        else (None if sample.should_refuse else 0.0),
        "faithfulness": normalized_faithfulness,
        "refusal_appropriate": normalized_refusal_appropriate,
        "notes": notes,
        "raw_content": raw_content,
    }


def load_eval_set(dataset_path: Path, limit: int | None = None) -> list[EvalSample]:
    rows: list[EvalSample] = []
    for line in dataset_path.read_text(encoding="utf-8-sig").splitlines():
        raw = line.strip()
        if not raw:
            continue
        data = json.loads(raw)
        rows.append(
            EvalSample(
                id=str(data["id"]),
                question=str(data["question"]),
                question_type=str(data["question_type"]),
                difficulty=str(data["difficulty"]),
                should_refuse=bool(data["should_refuse"]),
                gold_evidence=list(data.get("gold_evidence", [])),
                reference_answer=str(data.get("reference_answer", "")),
                key_points=[str(item) for item in data.get("key_points", [])],
                notes=str(data.get("notes", "")),
                tags=[str(item) for item in data.get("tags", [])],
            )
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def metadata_matches_evidence(metadata: dict[str, Any], evidence: dict[str, Any]) -> bool:
    if metadata.get("source") != evidence.get("source"):
        return False
    page_start = int(metadata.get("page_start", -1))
    page_end = int(metadata.get("page_end", -1))
    gold_start = int(evidence.get("page_start", -1))
    gold_end = int(evidence.get("page_end", -1))
    return not (page_end < gold_start or page_start > gold_end)


def compute_retrieval_metrics(
    sample: EvalSample,
    contexts: Sequence[dict[str, Any]],
) -> dict[str, float | None]:
    if sample.should_refuse:
        return {
            "hit_at_3": None,
            "mrr_at_5": None,
            "evidence_recall_at_5": None,
        }

    top3 = list(contexts[:3])
    top5 = list(contexts[:5])

    def hits(items: Sequence[dict[str, Any]], evidence: dict[str, Any]) -> bool:
        return any(metadata_matches_evidence(item["metadata"], evidence) for item in items)

    hit_at_3 = 1.0 if any(hits(top3, evidence) for evidence in sample.gold_evidence) else 0.0

    reciprocal_rank = 0.0
    for index, item in enumerate(top5, start=1):
        if any(metadata_matches_evidence(item["metadata"], evidence) for evidence in sample.gold_evidence):
            reciprocal_rank = 1.0 / index
            break

    evidence_recall = average(
        1.0 if hits(top5, evidence) else 0.0 for evidence in sample.gold_evidence
    )

    return {
        "hit_at_3": hit_at_3,
        "mrr_at_5": reciprocal_rank,
        "evidence_recall_at_5": evidence_recall,
    }


def build_context_preview(contexts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in contexts:
        meta = item["metadata"]
        snippet = item["text"].replace("\n", " ").strip()
        if len(snippet) > 280:
            snippet = snippet[:280].rstrip() + "..."
        items.append(
            {
                "source": meta["source"],
                "page_start": meta["page_start"],
                "page_end": meta["page_end"],
                "relevance_score": round_metric(item.get("relevance_score"), 4),
                "distance": round_metric(item.get("distance"), 4),
                "snippet": snippet,
            }
        )
    return items


def build_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def metric_bundle(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in records if item["sample"]["should_refuse"] is False]
    refusal_records = [item for item in records if item["sample"]["should_refuse"] is True]

    tp = sum(1 for item in records if item["sample"]["should_refuse"] and item["judge"]["predicted_decision"] == "refuse")
    fp = sum(1 for item in records if not item["sample"]["should_refuse"] and item["judge"]["predicted_decision"] == "refuse")
    fn = sum(1 for item in records if item["sample"]["should_refuse"] and item["judge"]["predicted_decision"] != "refuse")

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    refusal_f1 = safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None and (precision + recall) > 0 else 0.0

    return {
        "sample_count": len(records),
        "answerable_count": len(answerable),
        "refusal_count": len(refusal_records),
        "hit_at_3": average(item["retrieval"]["hit_at_3"] for item in answerable),
        "mrr_at_5": average(item["retrieval"]["mrr_at_5"] for item in answerable),
        "evidence_recall_at_5": average(item["retrieval"]["evidence_recall_at_5"] for item in answerable),
        "answer_accuracy": average(item["judge"]["answer_accuracy"] for item in answerable),
        "key_point_coverage": average(item["judge"]["key_point_coverage"] for item in answerable),
        "faithfulness": average(item["judge"]["faithfulness"] for item in answerable),
        "refusal_precision": precision,
        "refusal_recall": recall,
        "refusal_f1": refusal_f1,
        "over_refusal_rate": safe_div(fp, len(answerable)) if answerable else None,
        "refusal_appropriate_rate": average(
            item["judge"]["refusal_appropriate"] for item in refusal_records
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def build_breakdowns(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    by_difficulty: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        question_type = record["sample"]["question_type"]
        difficulty = record["sample"]["difficulty"]
        by_type.setdefault(question_type, []).append(record)
        by_difficulty.setdefault(difficulty, []).append(record)

    return {
        "question_type": {name: metric_bundle(group) for name, group in sorted(by_type.items())},
        "difficulty": {name: metric_bundle(group) for name, group in sorted(by_difficulty.items())},
    }


def sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            cleaned[key] = sanitize_metrics(value)
        elif isinstance(value, float):
            cleaned[key] = round_metric(value)
        else:
            cleaned[key] = value
    return cleaned


def render_summary_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# RAG Eval Summary",
        "",
        f"- run_time: `{summary['run_time']}`",
        f"- dataset: `{summary['dataset_path']}`",
        f"- sample_count: `{overall['sample_count']}`",
        f"- generation_model: `{summary['models']['generation_model']}`",
        f"- judge_model: `{summary['models']['judge_model']}`",
        "",
        "## Overall",
        "",
        f"- Hit@3: `{format_metric(overall['hit_at_3'])}`",
        f"- MRR@5: `{format_metric(overall['mrr_at_5'])}`",
        f"- Evidence Recall@5: `{format_metric(overall['evidence_recall_at_5'])}`",
        f"- Answer Accuracy: `{format_metric(overall['answer_accuracy'])}`",
        f"- Key Point Coverage: `{format_metric(overall['key_point_coverage'])}`",
        f"- Faithfulness: `{format_metric(overall['faithfulness'])}`",
        f"- Refusal F1: `{format_metric(overall['refusal_f1'])}`",
        f"- Over-Refusal Rate: `{format_metric(overall['over_refusal_rate'])}`",
        "",
        "## By Question Type",
        "",
    ]

    for name, metrics in summary["breakdowns"]["question_type"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- sample_count: `{metrics['sample_count']}`",
                f"- Hit@3: `{format_metric(metrics['hit_at_3'])}`",
                f"- Answer Accuracy: `{format_metric(metrics['answer_accuracy'])}`",
                f"- Faithfulness: `{format_metric(metrics['faithfulness'])}`",
                f"- Refusal F1: `{format_metric(metrics['refusal_f1'])}`",
                "",
            ]
        )

    lines.extend(["## By Difficulty", ""])
    for name, metrics in summary["breakdowns"]["difficulty"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- sample_count: `{metrics['sample_count']}`",
                f"- Hit@3: `{format_metric(metrics['hit_at_3'])}`",
                f"- Answer Accuracy: `{format_metric(metrics['answer_accuracy'])}`",
                f"- Faithfulness: `{format_metric(metrics['faithfulness'])}`",
                f"- Refusal F1: `{format_metric(metrics['refusal_f1'])}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isfinite(value):
        return f"{value:.4f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end RAG evaluation with judge scoring.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the eval JSONL dataset.",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Directory used to store evaluation runs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N samples for smoke testing.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when a sample fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    runs_dir = Path(args.runs_dir).resolve()

    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    samples = load_eval_set(dataset_path, limit=args.limit)
    if not samples:
        print("Error: no samples found in dataset.", file=sys.stderr)
        return 1

    rag_config = AppConfig.load()
    rag_config.retrieval_top_k = max(rag_config.retrieval_top_k, DEFAULT_RETRIEVAL_TOP_K)
    rag_config.rerank_top_n = max(rag_config.rerank_top_n, DEFAULT_RERANK_TOP_N)
    judge_config = JudgeConfig.load()

    runner = RAGRunner(rag_config)
    runner.ensure_ready()
    judge = JudgeClient(judge_config)

    run_dir = build_run_dir(runs_dir)
    per_sample_path = run_dir / "per_sample_results.jsonl"
    records: list[dict[str, Any]] = []

    print(f"Running evaluation on {len(samples)} samples...")
    print(f"Results will be saved to: {run_dir}")

    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {sample.id} | {sample.question}")
        try:
            contexts = runner.search(sample.question)
            answer = runner.answer(sample.question, contexts)
            retrieval = compute_retrieval_metrics(sample, contexts)
            judge_result = judge.evaluate(sample, contexts, answer)

            record = {
                "sample": asdict(sample),
                "retrieval": retrieval,
                "answer": answer,
                "judge": judge_result,
                "contexts": build_context_preview(contexts),
            }
            records.append(record)
            with per_sample_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                "  -> "
                f"decision={judge_result['predicted_decision']}, "
                f"accuracy={judge_result['answer_accuracy_score']}, "
                f"faithfulness={judge_result['faithfulness']}"
            )
        except Exception as exc:
            error_record = {
                "sample": asdict(sample),
                "error": str(exc),
            }
            with per_sample_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            print(f"  !! failed: {exc}", file=sys.stderr)
            if args.fail_fast:
                return 1

    successful_records = [item for item in records if "judge" in item]
    if not successful_records:
        print("Error: no samples completed successfully.", file=sys.stderr)
        return 1

    summary = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "output_dir": str(run_dir),
        "models": {
            "generation_model": rag_config.generation_model,
            "embedding_model": rag_config.embedding_model,
            "rerank_model": rag_config.rerank_model,
            "judge_model": judge_config.model,
        },
        "overall": metric_bundle(successful_records),
        "breakdowns": build_breakdowns(successful_records),
        "completed_samples": len(successful_records),
        "failed_samples": len(samples) - len(successful_records),
    }
    summary = sanitize_metrics(summary)

    summary_json_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md_path.write_text(render_summary_markdown(summary), encoding="utf-8")

    print("\nEvaluation completed.")
    print(f"Completed samples: {summary['completed_samples']}")
    print(f"Failed samples: {summary['failed_samples']}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary Markdown: {summary_md_path}")
    print("\nOverall metrics:")
    for key in [
        "hit_at_3",
        "mrr_at_5",
        "evidence_recall_at_5",
        "answer_accuracy",
        "key_point_coverage",
        "faithfulness",
        "refusal_f1",
        "over_refusal_rate",
    ]:
        print(f"- {key}: {format_metric(summary['overall'][key])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

