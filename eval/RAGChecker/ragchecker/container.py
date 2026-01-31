import json
from typing import List
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from . import metrics

AtomizedClaim = list[str] | tuple[str, ...] | str


def format_atomized_claim(claim: AtomizedClaim) -> str:
    """Convert a structured atomized claim into a readable string."""
    if isinstance(claim, (list, tuple)):
        return " | ".join(str(part) for part in claim)
    return str(claim)


def _collect_non_entailing_claims(
    claims: List[AtomizedClaim] | None,
    verdicts: List[str] | None,
) -> list[AtomizedClaim]:
    """Return the subset of claims whose verdict is not an entailment."""
    if not claims or not verdicts:
        return []

    false_claims: list[AtomizedClaim] = []
    for claim, verdict in zip(claims, verdicts):
        if isinstance(verdict, str) and verdict.lower() == "entailment":
            continue
        false_claims.append(claim)
    return false_claims


def _format_claim_section(title: str, claims: list[AtomizedClaim]) -> list[str]:
    """Format a section of claims for pretty-printing."""
    lines = [f"    {title}:"]
    if not claims:
        lines.append("      (none)")
        return lines

    for claim in claims:
        lines.append(f"      - {format_atomized_claim(claim)}")
    return lines


@dataclass_json
@dataclass
class RetrievedDoc:
    doc_id: str | None = None
    text: str = ""


@dataclass_json
@dataclass
class RAGResult:
    query_id: str
    query: str
    gt_answer: str
    response: str
    retrieved_context: List[RetrievedDoc] | None = None # Retrieved documents
    response_claims: List[List[str]] | None = None  # List of claims for the response
    gt_answer_claims: List[List[str]] | None = None  # List of claims for the ground truth answer
    answer2response: List[str] | None = None  # entailment results of answer -> response
    response2answer: List[str] | None = None  # entailment results of response -> answer
    retrieved2response: List[List[str]] | None = None  # entailment results of retrieved -> response
    retrieved2answer: List[List[str]] | None = None  # entailment results of retrieved -> answer
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def false_positive_atomized_claims(self) -> list[AtomizedClaim]:
        """Claims generated in the response that are not supported by the answer."""
        return _collect_non_entailing_claims(self.response_claims, self.answer2response)

    @property
    def false_negative_atomized_claims(self) -> list[AtomizedClaim]:
        """Ground truth claims that are not supported by the response."""
        return _collect_non_entailing_claims(self.gt_answer_claims, self.response2answer)


@dataclass_json
@dataclass
class RAGResults:
    results: List[RAGResult] = field(default_factory=list)
    metrics: dict[str, dict[str, float]] = field(default_factory = lambda: {
        metrics.overall_metrics: {},
        metrics.retriever_metrics: {},
        metrics.generator_metrics: {}
    })

    def __repr__(self) -> str:
        lines = [
            "RAGResults(",
            f"  {len(self.results):,} RAG results,",
        ]

        for result in self.results:
            lines.append(f"  Query ID: {result.query_id}")
            lines.extend(_format_claim_section(
                    "Atomized claims", result.response_claims
                )
            )
            lines.extend(
                _format_claim_section(
                    "False positive atomized claims", result.false_positive_atomized_claims
                )
            )
            lines.extend(
                _format_claim_section(
                    "False negative atomized claims", result.false_negative_atomized_claims
                )
            )

        lines.append("  Metrics:")
        for metric_line in json.dumps(self.metrics, indent=2).split("\n"):
            lines.append(f"    {metric_line}")
        lines.append(")")
        return "\n".join(lines)

    def update(self, rag_result: List[RAGResult]):
        self.results.append(rag_result)
        self.metrics = {
            metrics.overall_metrics: {},
            metrics.retriever_metrics: {},
            metrics.generator_metrics: {}
        }
