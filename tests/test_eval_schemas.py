"""Tests for evals.schemas — round-trip serialization for all three types."""

import json

from evals.schemas import GoldenThread, PredictionResult, RunMeta


class TestGoldenThread:
    def test_round_trip(self):
        gt = GoldenThread(
            thread_id="t_001",
            messages=[{"id": "msg_1", "payload": {"headers": []}}],
            senders=["Alice <alice@example.com>"],
            subject="Test subject",
            snippet="Test snippet",
            expected_sender_type="person",
            expected_label="needs_response",
            source="harvested",
            harvested_at="2024-01-01T00:00:00+00:00",
            reviewed=True,
            notes="Looks correct",
        )
        d = gt.to_dict()
        restored = GoldenThread.from_dict(d)
        assert restored.thread_id == gt.thread_id
        assert restored.messages == gt.messages
        assert restored.senders == gt.senders
        assert restored.subject == gt.subject
        assert restored.snippet == gt.snippet
        assert restored.expected_sender_type == gt.expected_sender_type
        assert restored.expected_label == gt.expected_label
        assert restored.source == gt.source
        assert restored.harvested_at == gt.harvested_at
        assert restored.reviewed == gt.reviewed
        assert restored.notes == gt.notes

    def test_round_trip_via_json(self):
        gt = GoldenThread(
            thread_id="t_002",
            messages=[{"id": "msg_2", "payload": {"headers": [], "body": {"data": "aGVsbG8="}}}],
            senders=["bob@example.com"],
            subject="JSON test",
            snippet="Testing JSON",
            expected_sender_type="service",
            expected_label="low_priority",
        )
        json_str = json.dumps(gt.to_dict())
        restored = GoldenThread.from_dict(json.loads(json_str))
        assert restored.thread_id == "t_002"
        assert restored.expected_sender_type == "service"
        assert restored.expected_label == "low_priority"
        assert restored.reviewed is False  # default
        assert restored.source == "harvested"  # default

    def test_defaults(self):
        d = {
            "thread_id": "t_003",
            "messages": [],
            "senders": [],
            "subject": "",
            "snippet": "",
            "expected_sender_type": "person",
            "expected_label": "fyi",
        }
        gt = GoldenThread.from_dict(d)
        assert gt.source == "harvested"
        assert gt.harvested_at == ""
        assert gt.reviewed is False
        assert gt.notes == ""
        assert gt.excluded is False

    def test_excluded_round_trips(self):
        gt = GoldenThread(
            thread_id="t_excl",
            messages=[],
            senders=[],
            subject="",
            snippet="",
            expected_sender_type="service",
            expected_label="low_priority",
            excluded=True,
        )
        d = gt.to_dict()
        assert d["excluded"] is True
        assert "skipped" not in d
        restored = GoldenThread.from_dict(d)
        assert restored.excluded is True

    def test_expected_assistant_defaults_to_none_and_round_trips(self):
        # Issue #78 groundwork: the annotation is tri-state. None = "not
        # annotated", which is what every pre-#78 record is.
        gt = GoldenThread(
            thread_id="t_asst",
            messages=[],
            senders=[],
            subject="",
            snippet="",
            expected_sender_type="person",
            expected_label="needs_response",
        )
        assert gt.expected_assistant is None
        gt.expected_assistant = True
        d = gt.to_dict()
        assert d["expected_assistant"] is True
        assert GoldenThread.from_dict(d).expected_assistant is True
        gt.expected_assistant = False
        assert GoldenThread.from_dict(gt.to_dict()).expected_assistant is False

    def test_records_without_expected_assistant_load_as_none(self):
        # Every record written before #78 lacks the key entirely.
        d = {
            "thread_id": "t_old",
            "messages": [],
            "senders": [],
            "subject": "",
            "snippet": "",
            "expected_sender_type": "person",
            "expected_label": "needs_response",
        }
        assert GoldenThread.from_dict(d).expected_assistant is None

    def test_legacy_skipped_key_maps_to_excluded(self):
        # Pre-existing golden sets persisted the marker as "skipped".
        d = {
            "thread_id": "t_legacy",
            "messages": [],
            "senders": [],
            "subject": "",
            "snippet": "",
            "expected_sender_type": "person",
            "expected_label": "fyi",
            "skipped": True,
        }
        gt = GoldenThread.from_dict(d)
        assert gt.excluded is True


class TestPredictionResult:
    def test_round_trip(self):
        pr = PredictionResult(
            thread_id="t_001",
            expected_sender_type="person",
            expected_label="needs_response",
            predicted_sender_type="person",
            predicted_label="needs_response",
            predicted_sender_type_raw="PERSON — this is a real human",
            predicted_label_raw="NEEDS_RESPONSE",
            sender_type_correct=True,
            label_correct=True,
            privacy_violation=False,
            duration_seconds=1.234,
            error=None,
        )
        d = pr.to_dict()
        assert d["type"] == "prediction"
        restored = PredictionResult.from_dict(d)
        assert restored.thread_id == pr.thread_id
        assert restored.predicted_sender_type == "person"
        assert restored.predicted_label == "needs_response"
        assert restored.sender_type_correct is True
        assert restored.label_correct is True
        assert restored.privacy_violation is False
        assert restored.duration_seconds == 1.234
        assert restored.error is None

    def test_assistant_fields_round_trip(self):
        # Issue #78 groundwork: the three assistant fields are carried through
        # the result record even though nothing predicts the field yet.
        pr = PredictionResult(
            thread_id="t_asst",
            expected_sender_type="person",
            expected_label="needs_response",
            expected_assistant=True,
            predicted_assistant=False,
            assistant_correct=False,
        )
        d = pr.to_dict()
        assert d["expected_assistant"] is True
        assert d["predicted_assistant"] is False
        assert d["assistant_correct"] is False
        restored = PredictionResult.from_dict(d)
        assert restored.expected_assistant is True
        assert restored.predicted_assistant is False
        assert restored.assistant_correct is False

    def test_assistant_fields_default_to_none(self):
        pr = PredictionResult(
            thread_id="t_plain",
            expected_sender_type="person",
            expected_label="fyi",
        )
        assert pr.expected_assistant is None
        assert pr.predicted_assistant is None
        assert pr.assistant_correct is None
        # A pre-#78 result record has none of the three keys.
        restored = PredictionResult.from_dict(
            {"thread_id": "t_old", "expected_sender_type": "person", "expected_label": "fyi"}
        )
        assert restored.expected_assistant is None
        assert restored.predicted_assistant is None
        assert restored.assistant_correct is None

    def test_round_trip_with_error(self):
        pr = PredictionResult(
            thread_id="t_err",
            expected_sender_type="service",
            expected_label="low_priority",
            error="Connection refused",
        )
        d = pr.to_dict()
        restored = PredictionResult.from_dict(d)
        assert restored.error == "Connection refused"
        assert restored.predicted_sender_type is None
        assert restored.predicted_label is None
        assert restored.sender_type_correct is None
        assert restored.label_correct is None

    def test_privacy_violation_flag(self):
        pr = PredictionResult(
            thread_id="t_priv",
            expected_sender_type="person",
            expected_label="fyi",
            predicted_sender_type="service",
            privacy_violation=True,
        )
        d = pr.to_dict()
        restored = PredictionResult.from_dict(d)
        assert restored.privacy_violation is True

    def test_round_trip_via_json(self):
        pr = PredictionResult(
            thread_id="t_json",
            expected_sender_type="service",
            expected_label="low_priority",
            predicted_sender_type="service",
            predicted_label="fyi",
            sender_type_correct=True,
            label_correct=False,
            duration_seconds=0.5,
        )
        json_str = json.dumps(pr.to_dict())
        restored = PredictionResult.from_dict(json.loads(json_str))
        assert restored.label_correct is False
        assert restored.sender_type_correct is True


class TestRunMeta:
    def test_round_trip(self):
        meta = RunMeta(
            run_id="abc123def456",
            timestamp="2024-01-15T10:30:00+00:00",
            config_hash="a1b2c3d4e5f6g7h8",
            config_path="config.toml",
            cloud_model="gpt-4",
            local_model="qwen3-32b",
            golden_set_path="evals/golden_set.jsonl",
            golden_set_count=50,
            stages="full",
            parallelism=3,
            tag="baseline",
            cloud_temperature=0.2,
            cloud_max_tokens=8096,
            cloud_extra_body={"top_p": 0.9},
            local_temperature=0.3,
            local_max_tokens=4096,
            local_extra_body=None,
            sender_system_prompt="Classify as PERSON or SERVICE",
            email_system_prompt="Classify the email",
            vip_email_system_prompt="Classify VIP email",
        )
        d = meta.to_dict()
        assert d["type"] == "run_meta"
        restored = RunMeta.from_dict(d)
        assert restored.run_id == meta.run_id
        assert restored.timestamp == meta.timestamp
        assert restored.config_hash == meta.config_hash
        assert restored.cloud_model == "gpt-4"
        assert restored.local_model == "qwen3-32b"
        assert restored.golden_set_count == 50
        assert restored.stages == "full"
        assert restored.parallelism == 3
        assert restored.tag == "baseline"
        assert restored.cloud_temperature == 0.2
        assert restored.cloud_max_tokens == 8096
        assert restored.cloud_extra_body == {"top_p": 0.9}
        assert restored.local_temperature == 0.3
        assert restored.local_max_tokens == 4096
        assert restored.local_extra_body is None
        assert restored.sender_system_prompt == "Classify as PERSON or SERVICE"
        assert restored.email_system_prompt == "Classify the email"
        assert restored.vip_email_system_prompt == "Classify VIP email"

    def test_defaults(self):
        d = {
            "run_id": "x",
            "timestamp": "t",
            "config_hash": "h",
            "config_path": "p",
            "cloud_model": "c",
            "local_model": "l",
            "golden_set_path": "g",
            "golden_set_count": 10,
        }
        meta = RunMeta.from_dict(d)
        assert meta.stages == "full"
        assert meta.parallelism == 1
        assert meta.tag == ""
        assert meta.cloud_temperature == 0.0
        assert meta.cloud_max_tokens == 0
        assert meta.cloud_extra_body is None
        assert meta.local_temperature == 0.0
        assert meta.local_max_tokens == 0
        assert meta.local_extra_body is None
        assert meta.sender_system_prompt == ""
        assert meta.email_system_prompt == ""
        assert meta.vip_email_system_prompt == ""

    def test_round_trip_via_json(self):
        meta = RunMeta(
            run_id="run1",
            timestamp="2024-06-01T00:00:00Z",
            config_hash="hash1",
            config_path="/path/to/config.toml",
            cloud_model="model-a",
            local_model="model-b",
            golden_set_path="gs.jsonl",
            golden_set_count=100,
            stages="stage1_only",
            parallelism=5,
            tag="experiment-1",
            cloud_temperature=0.5,
            cloud_max_tokens=2048,
            cloud_extra_body={"enable_thinking": False},
            local_temperature=0.1,
            local_max_tokens=1024,
            local_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            sender_system_prompt="sender prompt",
            email_system_prompt="email prompt",
            vip_email_system_prompt="vip prompt",
        )
        json_str = json.dumps(meta.to_dict())
        restored = RunMeta.from_dict(json.loads(json_str))
        assert restored.stages == "stage1_only"
        assert restored.parallelism == 5
        assert restored.tag == "experiment-1"
        assert restored.cloud_temperature == 0.5
        assert restored.cloud_max_tokens == 2048
        assert restored.cloud_extra_body == {"enable_thinking": False}
        assert restored.local_temperature == 0.1
        assert restored.local_max_tokens == 1024
        assert restored.local_extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
        assert restored.sender_system_prompt == "sender prompt"
        assert restored.email_system_prompt == "email prompt"
        assert restored.vip_email_system_prompt == "vip prompt"
