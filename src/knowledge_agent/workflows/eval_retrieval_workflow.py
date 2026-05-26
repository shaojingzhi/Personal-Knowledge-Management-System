from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from knowledge_agent.config import Settings
from knowledge_agent.storage.vector_store import IndexedQuestion, QuestionVectorStore
from knowledge_agent.workflows.retrieve_questions import QuestionRetriever

VALID_EVAL_MODES = ("vector", "bm25", "hybrid")


@dataclass(frozen=True)
class EvalSample:
    question: str
    expected_note_ids: list[str]
    expected_record_ids: list[str]
    expected_keywords: list[str]
    top_k: int


@dataclass(frozen=True)
class EvalRunResult:
    success: bool
    reason: str
    report_json_path: str | None
    report_markdown_path: str | None


class EvalRetrievalWorkflow:
    def __init__(self, settings: Settings, vector_store: QuestionVectorStore) -> None:
        self._settings = settings
        self._vector_store = vector_store

    def run(
        self,
        dataset_path: str,
        *,
        modes: list[str] | None = None,
    ) -> EvalRunResult:
        try:
            samples = self._load_samples(dataset_path)
            selected_modes = self._normalize_modes(modes)
            if not samples:
                return EvalRunResult(False, "Evaluation dataset is empty.", None, None)
            report = self._evaluate_samples(samples=samples, modes=selected_modes, dataset_path=dataset_path)
            report_dir = self._build_report_dir()
            report_dir.mkdir(parents=True, exist_ok=True)
            json_path = report_dir / "report.json"
            markdown_path = report_dir / "report.md"
            json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            markdown_path.write_text(self._build_markdown_report(report), encoding="utf-8")
            if not report["success"]:
                return EvalRunResult(
                    False,
                    "Retrieval evaluation completed with one or more mode failures.",
                    str(json_path),
                    str(markdown_path),
                )
            return EvalRunResult(True, "Retrieval evaluation completed.", str(json_path), str(markdown_path))
        except Exception as exc:
            return EvalRunResult(False, f"Retrieval evaluation failed: {exc}", None, None)

    def _load_samples(self, dataset_path: str) -> list[EvalSample]:
        payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError("Evaluation dataset must be a JSON array.")
        samples: list[EvalSample] = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise RuntimeError(f"Dataset row {index} must be a JSON object.")
            question = str(item.get("question", "")).strip()
            if not question:
                raise RuntimeError(f"Dataset row {index} is missing a non-empty question.")
            expected_note_ids = self._normalize_string_list(item.get("expected_note_ids"))
            expected_record_ids = self._normalize_string_list(item.get("expected_record_ids"))
            expected_keywords = self._normalize_string_list(item.get("expected_keywords"))
            if not expected_note_ids and not expected_record_ids and not expected_keywords:
                raise RuntimeError(
                    f"Dataset row {index} must define expected_note_ids, expected_record_ids, or expected_keywords."
                )
            samples.append(
                EvalSample(
                    question=question,
                    expected_note_ids=expected_note_ids,
                    expected_record_ids=expected_record_ids,
                    expected_keywords=expected_keywords,
                    top_k=self._normalize_top_k(item.get("top_k")),
                )
            )
        return samples

    def _normalize_modes(self, modes: list[str] | None) -> list[str]:
        if not modes:
            return list(VALID_EVAL_MODES)
        normalized = [mode.strip().lower() for mode in modes]
        invalid_modes = [mode for mode in normalized if mode not in VALID_EVAL_MODES]
        if invalid_modes:
            raise RuntimeError(f"Unsupported eval mode(s): {', '.join(invalid_modes)}")
        return normalized

    def _normalize_string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _normalize_top_k(self, value: object) -> int:
        if isinstance(value, int) and value > 0:
            return value
        return self._settings.retrieval_top_k

    def _evaluate_samples(self, *, samples: list[EvalSample], modes: list[str], dataset_path: str) -> dict[str, Any]:
        mode_reports = [self._evaluate_mode(mode=mode, samples=samples) for mode in modes]
        return {
            "dataset_path": dataset_path,
            "sample_count": len(samples),
            "generated_at": datetime.now().isoformat(),
            "success": all(bool(mode_report["success"]) for mode_report in mode_reports),
            "modes": mode_reports,
        }

    def _evaluate_mode(self, *, mode: str, samples: list[EvalSample]) -> dict[str, Any]:
        retriever = QuestionRetriever(self._settings, self._vector_store, retrieval_mode=mode)
        sample_reports: list[dict[str, Any]] = []
        mode_success = True
        recall_total = 0.0
        mrr_total = 0.0
        hit1_total = 0.0
        keyword_total = 0.0
        latency_total = 0.0

        for sample in samples:
            started = time.perf_counter()
            success, reason, results = retriever.search(sample.question, top_k=sample.top_k)
            latency_ms = (time.perf_counter() - started) * 1000
            latency_total += latency_ms
            sample_report = self._score_sample(sample=sample, success=success, reason=reason, results=results)
            if not success:
                mode_success = False
            sample_report["latency_ms"] = round(latency_ms, 2)
            sample_reports.append(sample_report)
            recall_total += sample_report["recall_at_k"]
            mrr_total += sample_report["mrr"]
            hit1_total += sample_report["hit_at_1"]
            keyword_total += sample_report["keyword_coverage"]

        sample_count = max(len(samples), 1)
        metrics = {
            "recall_at_k": round(recall_total / sample_count, 4),
            "mrr": round(mrr_total / sample_count, 4),
            "hit_at_1": round(hit1_total / sample_count, 4),
            "keyword_coverage": round(keyword_total / sample_count, 4),
            "avg_latency_ms": round(latency_total / sample_count, 2),
        }
        metrics["total_score"] = round(
            metrics["recall_at_k"] * 40
            + metrics["mrr"] * 20
            + metrics["hit_at_1"] * 20
            + metrics["keyword_coverage"] * 20,
            2,
        )
        return {
            "mode": mode,
            "success": mode_success,
            "metrics": metrics,
            "samples": sample_reports,
        }

    def _score_sample(
        self,
        *,
        sample: EvalSample,
        success: bool,
        reason: str,
        results: list[IndexedQuestion],
    ) -> dict[str, Any]:
        if not success:
            return {
                "question": sample.question,
                "success": False,
                "reason": reason,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "hit_at_1": 0.0,
                "keyword_coverage": 0.0,
                "results": [],
            }

        first_relevant_rank = self._first_relevant_rank(sample=sample, results=results)
        recall_at_k = self._recall_at_k(sample=sample, results=results)
        keyword_coverage = self._keyword_coverage(sample.expected_keywords, results)
        return {
            "question": sample.question,
            "success": True,
            "reason": reason,
            "recall_at_k": recall_at_k,
            "mrr": 0.0 if first_relevant_rank is None else round(1.0 / first_relevant_rank, 4),
            "hit_at_1": 1.0 if first_relevant_rank == 1 else 0.0,
            "keyword_coverage": keyword_coverage,
            "results": [
                {
                    "rank": index,
                    "record_id": item.record_id,
                    "note_id": item.note_id,
                    "score": round(item.score, 4),
                    "question": item.question,
                }
                for index, item in enumerate(results, start=1)
            ],
        }

    def _first_relevant_rank(self, *, sample: EvalSample, results: list[IndexedQuestion]) -> int | None:
        for index, item in enumerate(results, start=1):
            if sample.expected_record_ids and item.record_id in sample.expected_record_ids:
                return index
            if sample.expected_note_ids and item.note_id in sample.expected_note_ids:
                return index
        return None

    def _recall_at_k(self, *, sample: EvalSample, results: list[IndexedQuestion]) -> float:
        expected_targets = set(sample.expected_record_ids) | set(sample.expected_note_ids)
        if not expected_targets:
            return 1.0 if self._keyword_coverage(sample.expected_keywords, results) > 0 else 0.0
        matched_targets: set[str] = set()
        for item in results:
            if sample.expected_record_ids and item.record_id in sample.expected_record_ids:
                matched_targets.add(item.record_id)
            if sample.expected_note_ids and item.note_id in sample.expected_note_ids:
                matched_targets.add(item.note_id)
        return round(len(matched_targets) / len(expected_targets), 4)

    def _keyword_coverage(self, expected_keywords: list[str], results: list[IndexedQuestion]) -> float:
        if not expected_keywords:
            return 1.0
        joined = "\n".join(
            "\n".join(
                [
                    item.title,
                    item.summary,
                    item.question,
                    item.category,
                    " ".join(item.keywords),
                ]
            )
            for item in results
        )
        if not joined:
            return 0.0
        hit_count = sum(1 for keyword in expected_keywords if keyword.lower() in joined.lower())
        return round(hit_count / len(expected_keywords), 4)

    def _build_report_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return Path(self._settings.output_dir) / "evals" / f"retrieval-{timestamp}"

    def _build_markdown_report(self, report: dict[str, Any]) -> str:
        lines = [
            "# Retrieval Evaluation Report",
            "",
            f"- Dataset: `{report['dataset_path']}`",
            f"- Samples: {report['sample_count']}",
            f"- Generated At: {report['generated_at']}",
            "",
        ]
        for mode_report in report["modes"]:
            metrics = mode_report["metrics"]
            lines.extend(
                [
                    f"## Mode: {mode_report['mode']}",
                    "",
                    f"- Total Score: {metrics['total_score']}",
                    f"- Recall@K: {metrics['recall_at_k']}",
                    f"- MRR: {metrics['mrr']}",
                    f"- Hit@1: {metrics['hit_at_1']}",
                    f"- Keyword Coverage: {metrics['keyword_coverage']}",
                    f"- Avg Latency (ms): {metrics['avg_latency_ms']}",
                    "",
                ]
            )
        return "\n".join(lines) + "\n"
