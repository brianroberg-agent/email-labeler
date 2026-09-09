"""Dataclasses for golden set and evaluation results.

All types support JSONL serialization via to_dict() / from_dict().
"""

from dataclasses import dataclass


@dataclass
class GoldenThread:
    """One thread in the golden set (ground truth)."""

    thread_id: str
    messages: list[dict]  # Raw Gmail message resources from proxy_client.get_thread()
    senders: list[str]
    subject: str
    snippet: str
    expected_sender_type: str  # "person" or "service"
    expected_label: str  # "needs_response" / "fyi" / "low_priority"
    source: str = "harvested"  # "harvested" or "manual"
    harvested_at: str = ""  # ISO 8601
    reviewed: bool = False
    notes: str = ""
    excluded: bool = False  # permanently set aside: never reviewed-queued or evaluated
    # Tri-state annotation (issue #78): True/False = an obligation the assistant
    # can/cannot discharge; None = not annotated. Meaningful only while
    # expected_label is "needs_response" — the review tools clear it when the
    # label moves away.
    expected_assistant: bool | None = None

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "messages": self.messages,
            "senders": self.senders,
            "subject": self.subject,
            "snippet": self.snippet,
            "expected_sender_type": self.expected_sender_type,
            "expected_label": self.expected_label,
            "source": self.source,
            "harvested_at": self.harvested_at,
            "reviewed": self.reviewed,
            "notes": self.notes,
            "excluded": self.excluded,
            "expected_assistant": self.expected_assistant,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenThread":
        return cls(
            thread_id=d["thread_id"],
            messages=d["messages"],
            senders=d["senders"],
            subject=d["subject"],
            snippet=d["snippet"],
            expected_sender_type=d["expected_sender_type"],
            expected_label=d["expected_label"],
            source=d.get("source", "harvested"),
            harvested_at=d.get("harvested_at", ""),
            reviewed=d.get("reviewed", False),
            notes=d.get("notes", ""),
            # Backward compat: pre-existing golden sets persisted this as "skipped".
            excluded=d.get("excluded", d.get("skipped", False)),
            # Absent on every record written before #78.
            expected_assistant=d.get("expected_assistant"),
        )


@dataclass
class PredictionResult:
    """One prediction from an evaluation run."""

    thread_id: str
    expected_sender_type: str
    expected_label: str
    predicted_sender_type: str | None = None
    predicted_label: str | None = None
    predicted_sender_type_raw: str | None = None
    predicted_label_raw: str | None = None
    sender_type_correct: bool | None = None
    label_correct: bool | None = None
    # Assistant field (issue #78). expected_ mirrors the golden annotation;
    # predicted_ is None until something predicts it; assistant_correct is the
    # per-thread comparison, None whenever either side is missing.
    expected_assistant: bool | None = None
    predicted_assistant: bool | None = None
    assistant_correct: bool | None = None
    privacy_violation: bool = False  # True if expected=person, predicted=service
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "type": "prediction",
            "thread_id": self.thread_id,
            "expected_sender_type": self.expected_sender_type,
            "expected_label": self.expected_label,
            "predicted_sender_type": self.predicted_sender_type,
            "predicted_label": self.predicted_label,
            "predicted_sender_type_raw": self.predicted_sender_type_raw,
            "predicted_label_raw": self.predicted_label_raw,
            "sender_type_correct": self.sender_type_correct,
            "label_correct": self.label_correct,
            "expected_assistant": self.expected_assistant,
            "predicted_assistant": self.predicted_assistant,
            "assistant_correct": self.assistant_correct,
            "privacy_violation": self.privacy_violation,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionResult":
        return cls(
            thread_id=d["thread_id"],
            expected_sender_type=d["expected_sender_type"],
            expected_label=d["expected_label"],
            predicted_sender_type=d.get("predicted_sender_type"),
            predicted_label=d.get("predicted_label"),
            predicted_sender_type_raw=d.get("predicted_sender_type_raw"),
            predicted_label_raw=d.get("predicted_label_raw"),
            sender_type_correct=d.get("sender_type_correct"),
            label_correct=d.get("label_correct"),
            expected_assistant=d.get("expected_assistant"),
            predicted_assistant=d.get("predicted_assistant"),
            assistant_correct=d.get("assistant_correct"),
            privacy_violation=d.get("privacy_violation", False),
            duration_seconds=d.get("duration_seconds", 0.0),
            error=d.get("error"),
        )


@dataclass
class ThinkingEntry:
    """Chain-of-thought content for one thread in an eval run.

    Stored in sidecar .cot.jsonl files alongside main results.
    """

    thread_id: str
    stage1_thinking: str = ""
    stage2_thinking: str = ""

    def to_dict(self) -> dict:
        return {
            "type": "thinking",
            "thread_id": self.thread_id,
            "stage1_thinking": self.stage1_thinking,
            "stage2_thinking": self.stage2_thinking,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThinkingEntry":
        return cls(
            thread_id=d["thread_id"],
            stage1_thinking=d.get("stage1_thinking", ""),
            stage2_thinking=d.get("stage2_thinking", ""),
        )


@dataclass
class RunMeta:
    """Metadata for an evaluation run (first line of results JSONL)."""

    run_id: str
    timestamp: str  # ISO 8601
    config_hash: str
    config_path: str
    cloud_model: str
    local_model: str
    golden_set_path: str
    golden_set_count: int
    stages: str = "full"  # "full" / "stage1_only" / "stage2_only"
    parallelism: int = 1
    tag: str = ""
    # Cache key parameters (per-LLM)
    cloud_temperature: float = 0.0
    cloud_max_tokens: int = 0
    cloud_extra_body: dict | None = None
    local_temperature: float = 0.0
    local_max_tokens: int = 0
    local_extra_body: dict | None = None
    # System prompts (constant across a run, deterministic from config)
    sender_system_prompt: str = ""
    email_system_prompt: str = ""
    vip_email_system_prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "type": "run_meta",
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "config_path": self.config_path,
            "cloud_model": self.cloud_model,
            "local_model": self.local_model,
            "golden_set_path": self.golden_set_path,
            "golden_set_count": self.golden_set_count,
            "stages": self.stages,
            "parallelism": self.parallelism,
            "tag": self.tag,
            "cloud_temperature": self.cloud_temperature,
            "cloud_max_tokens": self.cloud_max_tokens,
            "cloud_extra_body": self.cloud_extra_body,
            "local_temperature": self.local_temperature,
            "local_max_tokens": self.local_max_tokens,
            "local_extra_body": self.local_extra_body,
            "sender_system_prompt": self.sender_system_prompt,
            "email_system_prompt": self.email_system_prompt,
            "vip_email_system_prompt": self.vip_email_system_prompt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunMeta":
        return cls(
            run_id=d["run_id"],
            timestamp=d["timestamp"],
            config_hash=d["config_hash"],
            config_path=d["config_path"],
            cloud_model=d["cloud_model"],
            local_model=d["local_model"],
            golden_set_path=d["golden_set_path"],
            golden_set_count=d["golden_set_count"],
            stages=d.get("stages", "full"),
            parallelism=d.get("parallelism", 1),
            tag=d.get("tag", ""),
            cloud_temperature=d.get("cloud_temperature", 0.0),
            cloud_max_tokens=d.get("cloud_max_tokens", 0),
            cloud_extra_body=d.get("cloud_extra_body"),
            local_temperature=d.get("local_temperature", 0.0),
            local_max_tokens=d.get("local_max_tokens", 0),
            local_extra_body=d.get("local_extra_body"),
            sender_system_prompt=d.get("sender_system_prompt", ""),
            email_system_prompt=d.get("email_system_prompt", ""),
            vip_email_system_prompt=d.get("vip_email_system_prompt", ""),
        )
