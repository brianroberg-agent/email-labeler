"""Tests for eval runner parallelism defaults and extra_body overrides."""

import argparse
import asyncio
import json

import pytest

from daemon import DEFAULT_MAX_THREAD_CHARS, load_config
from evals import run_eval
from evals.run_eval import (
    apply_config_overrides,
    default_tag,
    load_golden_set,
    main,
    maybe_report,
    preflight_check,
    required_endpoints,
    resolve_extra_body,
    resolve_local_only,
    resolve_max_thread_chars,
    resolve_parallelism,
    sanitize_tag,
    write_results,
)
from evals.schemas import GoldenThread, PredictionResult, RunMeta


def _golden(thread_id, **kw):
    base = dict(
        thread_id=thread_id, messages=[], senders=[], subject="", snippet="",
        expected_sender_type="person", expected_label="fyi",
    )
    base.update(kw)
    return GoldenThread(**base)


def _write_golden_set(path, threads):
    path.write_text("".join(json.dumps(t.to_dict()) + "\n" for t in threads))


def _override_args(**kw):
    """argparse.Namespace with every override field defaulting to None (not provided)."""
    base = dict(
        cloud_model=None, cloud_temperature=None, cloud_max_tokens=None, cloud_timeout=None,
        local_model=None, local_temperature=None, local_max_tokens=None, local_timeout=None,
        cloud_no_think=False, cloud_extra_body=None,
        local_no_think=False, local_extra_body=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _override_config():
    return {"llm": {
        "cloud": {"model": "cloud-default", "temperature": 0.5, "max_tokens": 100, "timeout": 60},
        "local": {"model": "local-default", "temperature": 0.7, "max_tokens": 200, "timeout": 180},
    }}


class TestLoadGoldenSet:
    def test_excluded_threads_are_dropped(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        _write_golden_set(path, [
            _golden("keep", reviewed=True),
            _golden("drop", reviewed=True, excluded=True),
        ])
        loaded = load_golden_set(path)
        assert [t.thread_id for t in loaded] == ["keep"]

    def test_excluded_dropped_even_when_unreviewed_included(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        _write_golden_set(path, [
            _golden("keep", reviewed=False),
            _golden("drop", reviewed=False, excluded=True),
        ])
        loaded = load_golden_set(path, reviewed_only=False)
        assert [t.thread_id for t in loaded] == ["keep"]


class TestApplyConfigOverrides:
    def test_no_overrides_leaves_config_unchanged(self):
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args())
        assert cfg == _override_config()

    def test_local_timeout_override(self):
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(local_timeout=900))
        assert cfg["llm"]["local"]["timeout"] == 900
        assert cfg["llm"]["cloud"]["timeout"] == 60  # cloud untouched

    def test_cloud_timeout_override(self):
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(cloud_timeout=45))
        assert cfg["llm"]["cloud"]["timeout"] == 45
        assert cfg["llm"]["local"]["timeout"] == 180  # local untouched

    def test_timeout_override_does_not_disturb_other_fields(self):
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(local_timeout=900))
        assert cfg["llm"]["local"]["model"] == "local-default"
        assert cfg["llm"]["local"]["max_tokens"] == 200

    def test_zero_is_a_valid_override(self):
        # 0.0 is not None, so it must override even though it's falsy.
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(local_temperature=0.0))
        assert cfg["llm"]["local"]["temperature"] == 0.0

    def test_multiple_overrides_at_once(self):
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(
            local_model="m", local_max_tokens=512, local_timeout=600, cloud_timeout=30))
        assert cfg["llm"]["local"]["model"] == "m"
        assert cfg["llm"]["local"]["max_tokens"] == 512
        assert cfg["llm"]["local"]["timeout"] == 600
        assert cfg["llm"]["cloud"]["timeout"] == 30

    def test_no_think_applies_extra_body_in_one_pass(self):
        # extra_body overrides are folded into apply_config_overrides, not a second
        # parallel loop in main() (review finding #10).
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(local_no_think=True))
        assert cfg["llm"]["local"]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }
        assert "extra_body" not in cfg["llm"]["cloud"]  # cloud untouched

    def test_cloud_no_think_applies_both_dialects_to_cloud(self):
        # The cloud flag's wiring is separate from the local flag's — a
        # disconnected --cloud-no-think would be a silently-inert no-op, the
        # exact failure shape issue #64 is about.
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(cloud_no_think=True))
        assert cfg["llm"]["cloud"]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }
        assert "extra_body" not in cfg["llm"]["local"]  # local untouched

    def test_extra_body_json_is_merged(self):
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(cloud_extra_body='{"top_p": 0.9}'))
        assert cfg["llm"]["cloud"]["extra_body"] == {"top_p": 0.9}

    def test_invalid_extra_body_json_raises_valueerror(self):
        cfg = _override_config()
        with pytest.raises(ValueError):
            apply_config_overrides(cfg, _override_args(local_extra_body="{not json}"))


class TestResolveMaxThreadChars:
    def test_uses_config_value_when_present(self):
        assert resolve_max_thread_chars({"daemon": {"max_thread_chars": 12345}}) == 12345

    def test_falls_back_to_shared_daemon_default(self):
        assert resolve_max_thread_chars({"daemon": {}}) == DEFAULT_MAX_THREAD_CHARS
        assert resolve_max_thread_chars({}) == DEFAULT_MAX_THREAD_CHARS

    def test_default_is_not_the_stale_50000(self):
        # Regression (finding #9): the eval fallback drifted to 50000 while the
        # daemon/config canonical default moved to 16000.
        assert resolve_max_thread_chars({}) == 16000


class TestResolveParallelism:
    def test_cli_override_wins(self):
        """Explicit --parallelism on CLI takes precedence over config."""
        config = load_config()
        assert resolve_parallelism(5, config) == 5

    def test_none_falls_back_to_config(self):
        """When --parallelism is not specified, use cloud_parallel from config."""
        config = load_config()
        expected = config["daemon"]["cloud_parallel"]
        assert resolve_parallelism(None, config) == expected

    def test_none_falls_back_to_default_without_config_key(self):
        """When config has no cloud_parallel, fall back to 1."""
        config = {"daemon": {}}
        assert resolve_parallelism(None, config) == 1


class TestResolveExtraBody:
    def test_none_when_nothing_set(self):
        assert resolve_extra_body(None, False, None) is None

    def test_passes_through_base_when_no_overrides(self):
        assert resolve_extra_body({"top_p": 0.9}, False, None) == {"top_p": 0.9}

    def test_no_think_sets_both_disable_dialects(self):
        # Issue #64: --no-think used to emit only the chat_template_kwargs form,
        # which Ollama silently ignores — so a think-on/think-off A/B against an
        # Ollama backend compared think-on to think-on. The flag must emit BOTH
        # known dialects: chat_template_kwargs.enable_thinking=false (mlx_lm /
        # LM Studio) and top-level reasoning_effort="none" (Ollama; only "none"
        # disables there).
        assert resolve_extra_body(None, True, None) == {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }

    def test_no_think_preserves_existing_chat_template_kwargs(self):
        out = resolve_extra_body({"chat_template_kwargs": {"foo": 1}}, True, None)
        assert out == {
            "chat_template_kwargs": {"foo": 1, "enable_thinking": False},
            "reasoning_effort": "none",
        }

    def test_explicit_json_merges_and_wins(self):
        out = resolve_extra_body({"top_p": 0.9}, False, '{"top_p": 0.5, "min_p": 0.1}')
        assert out == {"top_p": 0.5, "min_p": 0.1}

    def test_explicit_json_overrides_no_think(self):
        # JSON overrides the keys it names; --no-think keys it doesn't name stay.
        out = resolve_extra_body(None, True, '{"chat_template_kwargs": {"enable_thinking": true}}')
        assert out == {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_effort": "none",
        }

    def test_explicit_json_overrides_no_think_reasoning_effort(self):
        out = resolve_extra_body(None, True, '{"reasoning_effort": "high"}')
        assert out == {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "high",
        }

    def test_no_think_overrides_base_config_reasoning_effort(self):
        # Precedence: --no-think must OVERRIDE a base config's reasoning_effort,
        # not merely fill it in when absent. A base carrying "low" is plausible
        # Ollama tuning — and "low" is a silent no-op there (thinking stays ON),
        # so preserving it would make --no-think inert against exactly the
        # backend it was fixed for.
        out = resolve_extra_body({"reasoning_effort": "low"}, True, None)
        assert out["reasoning_effort"] == "none"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            resolve_extra_body(None, False, "{not json}")

    def test_non_object_json_raises(self):
        with pytest.raises(ValueError):
            resolve_extra_body(None, False, "[1, 2, 3]")

    def test_does_not_mutate_base(self):
        base = {"chat_template_kwargs": {"foo": 1}}
        resolve_extra_body(base, True, None)
        assert base == {"chat_template_kwargs": {"foo": 1}}


class TestResolveLocalOnly:
    """--local-only is shorthand for --stages stage2_only --sender-type person,
    the only run configuration that exercises ONLY the local classifier."""

    def _args(self, **kw):
        base = dict(local_only=False, stages="full", sender_type=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_noop_when_not_set(self):
        args = self._args(local_only=False, stages="full", sender_type=None)
        resolve_local_only(args)
        assert args.stages == "full"
        assert args.sender_type is None

    def test_sets_stage2_person_when_set(self):
        args = self._args(local_only=True)
        resolve_local_only(args)
        assert args.stages == "stage2_only"
        assert args.sender_type == "person"

    def test_compatible_explicit_values_allowed(self):
        args = self._args(local_only=True, stages="stage2_only", sender_type="person")
        resolve_local_only(args)
        assert (args.stages, args.sender_type) == ("stage2_only", "person")

    def test_conflicting_stages_raises(self):
        args = self._args(local_only=True, stages="stage1_only")
        with pytest.raises(ValueError, match="local-only"):
            resolve_local_only(args)

    def test_conflicting_sender_type_raises(self):
        args = self._args(local_only=True, sender_type="service")
        with pytest.raises(ValueError, match="local-only"):
            resolve_local_only(args)


class TestSanitizeTag:
    def test_replaces_slashes_in_hf_id(self):
        assert sanitize_tag("qwen/qwen3-14b") == "qwen-qwen3-14b"

    def test_keeps_dots_dashes_underscores(self):
        assert sanitize_tag("Qwen3-14B_v2.1") == "Qwen3-14B_v2.1"

    def test_collapses_runs_and_strips_edges(self):
        assert sanitize_tag("/foo //bar/") == "foo-bar"

    def test_empty_stays_empty(self):
        assert sanitize_tag("") == ""


class TestDefaultTag:
    def _config(self):
        return {"llm": {
            "cloud": {"model": "anthropic/claude-x"},
            "local": {"model": "qwen/qwen3-14b"},
        }}

    def test_uses_local_model_for_stage2_only(self):
        assert default_tag("stage2_only", self._config()) == "qwen-qwen3-14b"

    def test_uses_local_model_for_full(self):
        assert default_tag("full", self._config()) == "qwen-qwen3-14b"

    def test_uses_cloud_model_for_stage1_only(self):
        # Stage 1 is the cloud model's job; tagging by local model would mislabel.
        assert default_tag("stage1_only", self._config()) == "anthropic-claude-x"


class TestRequiredEndpoints:
    """Endpoints WITHOUT which a run produces no results at all, so preflight
    only hard-aborts when the whole run would be useless (not when one endpoint
    is down but the other half of the work could still succeed)."""

    def test_stage1_only_needs_cloud_only(self):
        assert required_endpoints("stage1_only", None) == (True, False)

    def test_stage2_only_person_needs_local_only(self):
        assert required_endpoints("stage2_only", "person") == (False, True)

    def test_stage2_only_service_needs_cloud_only(self):
        assert required_endpoints("stage2_only", "service") == (True, False)

    def test_stage2_only_mixed_requires_neither(self):
        # person->local, service->cloud: a down endpoint loses half the work,
        # not all of it, so neither is wholly required.
        assert required_endpoints("stage2_only", None) == (False, False)

    def test_full_requires_cloud_only(self):
        # Stage 1 (cloud) gates every thread, so cloud-down means zero results;
        # local-down only fails person Stage 2b, leaving partial cloud results.
        assert required_endpoints("full", "person") == (True, False)
        assert required_endpoints("full", None) == (True, False)


class _FakeClient:
    def __init__(self, model, base_url, available, status_code=None, error=None):
        self.model = model
        self.base_url = base_url
        self._available = available
        self._status_code = status_code
        self._error = error
        self.checked = False
        self.last_timeout = "unset"

    async def is_available(self, timeout=None):
        return (await self.probe(timeout)).ok

    async def probe(self, timeout=None):
        from llm_client import AvailabilityResult
        self.checked = True
        self.last_timeout = timeout
        return AvailabilityResult(
            ok=self._available, status_code=self._status_code, error=self._error
        )


def _run(coro):
    return asyncio.run(coro)


class TestPreflightCheck:
    def test_no_errors_when_required_endpoints_reachable(self):
        cloud = _FakeClient("c", "http://cloud", True)
        local = _FakeClient("m", "http://local", True)
        assert _run(preflight_check(cloud, local, True, True)) == []

    def test_unreachable_required_local_reported(self):
        cloud = _FakeClient("c", "http://cloud", True)
        local = _FakeClient("qwen3", "http://local", False)
        errors = _run(preflight_check(cloud, local, False, True))
        assert len(errors) == 1
        assert "qwen3" in errors[0] and "http://local" in errors[0]

    def test_unreachable_required_cloud_reported(self):
        cloud = _FakeClient("claude", "http://cloud", False)
        local = _FakeClient("m", "http://local", True)
        errors = _run(preflight_check(cloud, local, True, True))
        assert len(errors) == 1
        assert "claude" in errors[0]

    def test_unneeded_endpoint_not_checked_even_if_down(self):
        cloud = _FakeClient("c", "http://cloud", False)
        local = _FakeClient("m", "http://local", True)
        errors = _run(preflight_check(cloud, local, False, True))
        assert errors == []
        assert cloud.checked is False  # never probed

    def test_404_shows_model_mismatch_hint(self):
        # Issue #41 item 7: the "model-name mismatch returns 404" hint appears
        # only when the probe actually got a 404.
        cloud = _FakeClient("c", "http://cloud", True)
        local = _FakeClient("qwen3", "http://local", False, status_code=404)
        errors = _run(preflight_check(cloud, local, False, True))
        assert len(errors) == 1
        assert "404" in errors[0]

    def test_non_404_omits_model_mismatch_hint(self):
        cloud = _FakeClient("c", "http://cloud", True)
        local = _FakeClient("qwen3", "http://local", False, status_code=503)
        errors = _run(preflight_check(cloud, local, False, True))
        assert len(errors) == 1
        assert "404" not in errors[0]

    def test_unreachable_no_response_omits_hint(self):
        # No HTTP response at all (status_code None) -> no 404 hint.
        cloud = _FakeClient("c", "http://cloud", True)
        local = _FakeClient("qwen3", "http://local", False)
        errors = _run(preflight_check(cloud, local, False, True))
        assert "404" not in errors[0]

    def test_non_404_status_code_is_shown(self):
        # Finding 1: a non-404 status (e.g. 401 bad key, 500 down) must be
        # surfaced, not discarded, so the user can tell them apart.
        cloud = _FakeClient("c", "http://cloud", True)
        local = _FakeClient("qwen3", "http://local", False, status_code=401)
        errors = _run(preflight_check(cloud, local, False, True))
        assert len(errors) == 1
        assert "401" in errors[0]

    def test_connection_error_text_is_shown(self):
        # Finding 1: probe()'s captured exception text (connect/timeout/
        # UnsupportedProtocol) must appear so a down server / bad URL is
        # distinguishable from a bad key.
        cloud = _FakeClient("c", "http://cloud", True)
        local = _FakeClient(
            "qwen3", "http://local", False, error="ConnectError: Connection refused"
        )
        errors = _run(preflight_check(cloud, local, False, True))
        assert len(errors) == 1
        assert "ConnectError" in errors[0]

    def test_forwards_per_endpoint_timeouts_to_is_available(self):
        # Cloud and local get their own probe timeouts (cloud need not wait out
        # the long local cold-load budget).
        cloud = _FakeClient("c", "http://cloud", True)
        local = _FakeClient("m", "http://local", True)
        _run(preflight_check(cloud, local, True, True, cloud_timeout=30, local_timeout=180))
        assert cloud.last_timeout == 30
        assert local.last_timeout == 180


def _meta(**overrides):
    defaults = dict(
        run_id="abc12345-6789", timestamp="2025-01-01T00:00:00", config_hash="deadbeef",
        config_path="config.toml", cloud_model="test-cloud", local_model="test-local",
        golden_set_path="golden.jsonl", golden_set_count=1,
    )
    defaults.update(overrides)
    return RunMeta(**defaults)


def _result(predicted_label="fyi"):
    return PredictionResult(
        thread_id="t1", expected_sender_type="person", expected_label="fyi",
        predicted_sender_type="person", predicted_label=predicted_label,
        sender_type_correct=True, label_correct=(predicted_label == "fyi"),
    )


class TestMaybeReport:
    def test_prints_nothing_when_neither_flag_set(self, capsys):
        maybe_report(_meta(), [_result()], report_enabled=False, compare_to=None)
        assert capsys.readouterr().out == ""

    def test_report_flag_prints_evaluation_report(self, capsys):
        maybe_report(_meta(), [_result()], report_enabled=True, compare_to=None)
        assert "Evaluation Report" in capsys.readouterr().out

    def test_compare_to_prints_comparison(self, capsys, tmp_path):
        prior = tmp_path / "20250101_000000_stage2_only_prior_aaaa1111.jsonl"
        write_results(prior, _meta(run_id="prior999", tag="prior"), [_result()])
        maybe_report(_meta(), [_result()], report_enabled=False, compare_to=str(prior))
        assert "Comparison" in capsys.readouterr().out

    def test_missing_compare_file_errors_without_raising(self, capsys, tmp_path):
        missing = tmp_path / "nope.jsonl"
        maybe_report(_meta(), [_result()], report_enabled=False, compare_to=str(missing))
        err = capsys.readouterr().err
        assert "not found" in err

    def test_baseline_is_run_a_so_delta_reads_new_minus_baseline(self, capsys, tmp_path):
        prior = tmp_path / "20250101_000000_stage2_only_base_aaaa1111.jsonl"
        write_results(prior, _meta(run_id="prior999", tag="base"), [_result()])
        maybe_report(_meta(run_id="new00000", tag="new"), [_result()],
                     report_enabled=False, compare_to=str(prior))
        out = capsys.readouterr().out
        a_line = next(ln for ln in out.splitlines() if ln.strip().startswith("Run A:"))
        b_line = next(ln for ln in out.splitlines() if ln.strip().startswith("Run B:"))
        assert "base" in a_line  # baseline is A
        assert "new" in b_line   # this run is B -> Delta = new - baseline

    def test_compare_shows_per_thread_differences(self, capsys, tmp_path):
        prior = tmp_path / "20250101_000000_stage2_only_base_aaaa1111.jsonl"
        write_results(prior, _meta(tag="base"), [_result(predicted_label="fyi")])
        maybe_report(_meta(tag="new"), [_result(predicted_label="low_priority")],
                     report_enabled=False, compare_to=str(prior))
        assert "Prediction Differences" in capsys.readouterr().out

    def test_malformed_compare_file_errors_without_raising(self, capsys, tmp_path):
        bad = tmp_path / "20250101_000000_stage2_only_base_aaaa1111.jsonl"
        bad.write_text('{"type": "thinking", "thread_id": "t1"}\n')  # no run_meta line
        maybe_report(_meta(), [_result()], report_enabled=False, compare_to=str(bad))
        assert "Error" in capsys.readouterr().err

    def test_population_mismatch_warns(self, capsys, tmp_path):
        prior = tmp_path / "20250101_000000_full_base_aaaa1111.jsonl"
        write_results(prior, _meta(tag="base", stages="full"), [_result()])
        maybe_report(_meta(tag="new", stages="stage2_only"), [_result()],
                     report_enabled=False, compare_to=str(prior))
        assert "stages" in capsys.readouterr().err.lower()


def _full_args(**kw):
    """A complete run_eval argparse.Namespace with CLI defaults, overridable per test."""
    base = dict(
        golden_set="evals/golden_set.jsonl", config=None, output_dir="evals/results/",
        stages="full", parallelism=None, include_unreviewed=False, dry_run=False,
        tag=None, no_cache=True, sender_type=None, max_threads=None,
        local_only=False, skip_preflight=False, preflight_timeout=None,
        report=False, compare_to=None,
        cloud_model=None, local_model=None,
        cloud_temperature=None, local_temperature=None,
        cloud_max_tokens=None, local_max_tokens=None,
        cloud_timeout=None, local_timeout=None,
        cloud_extra_body=None, local_extra_body=None,
        cloud_no_think=False, local_no_think=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


class TestMainLocalOnlyWiring:
    def test_local_only_dry_run_filters_to_person_and_defaults_tag(
        self, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setenv("MLX_MODEL", "qwen/qwen3-test")
        path = tmp_path / "golden.jsonl"
        _write_golden_set(path, [
            _golden("person_thread", reviewed=True, expected_sender_type="person"),
            _golden("service_thread", reviewed=True, expected_sender_type="service"),
        ])
        args = _full_args(golden_set=str(path), local_only=True, dry_run=True)

        asyncio.run(main(args))

        # --local-only resolved the run shape in place...
        assert args.stages == "stage2_only"
        assert args.sender_type == "person"
        # ...the tag defaulted to the (sanitized) local model under test...
        assert args.tag == "qwen-qwen3-test"
        # ...and the service thread was filtered out of the dry-run preview.
        err = capsys.readouterr().err
        assert "person_thread" in err
        assert "service_thread" not in err


class TestMainPreflightWiring:
    def test_unreachable_local_endpoint_exits_1_before_evaluating(
        self, tmp_path, monkeypatch
    ):
        async def _down(self, timeout=None):
            from llm_client import AvailabilityResult
            return AvailabilityResult(ok=False)

        monkeypatch.setattr(run_eval.LLMClient, "probe", _down)
        path = tmp_path / "golden.jsonl"
        _write_golden_set(path, [
            _golden("person_thread", reviewed=True, expected_sender_type="person"),
        ])
        # local-only -> only the local endpoint is required; it's "down" -> exit 1.
        args = _full_args(golden_set=str(path), local_only=True)

        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(main(args))
        assert excinfo.value.code == 1

    def test_skip_preflight_bypasses_the_check(self, tmp_path, monkeypatch):
        # With the endpoint "down" but --skip-preflight, main must get PAST preflight
        # (and then fail later trying to actually classify — proving preflight was skipped).
        async def _down(self, timeout=None):
            from llm_client import AvailabilityResult
            return AvailabilityResult(ok=False)

        async def _boom(*a, **k):
            raise RuntimeError("reached evaluation")

        monkeypatch.setattr(run_eval.LLMClient, "probe", _down)
        monkeypatch.setattr(run_eval, "run_evaluation", _boom)
        path = tmp_path / "golden.jsonl"
        _write_golden_set(path, [
            _golden("person_thread", reviewed=True, expected_sender_type="person"),
        ])
        args = _full_args(golden_set=str(path), local_only=True, skip_preflight=True)

        with pytest.raises(RuntimeError, match="reached evaluation"):
            asyncio.run(main(args))


class TestMainPreflightTimeoutWiring:
    def _capture_and_stop(self, monkeypatch):
        """Patch preflight_check to capture its timeouts and run_evaluation to stop main."""
        captured = {}

        async def fake_preflight(cloud, local, need_cloud, need_local,
                                 cloud_timeout=None, local_timeout=None):
            captured["cloud_timeout"] = cloud_timeout
            captured["local_timeout"] = local_timeout
            return []  # reachable

        async def boom(*a, **k):
            raise RuntimeError("reached evaluation")

        monkeypatch.setattr(run_eval, "preflight_check", fake_preflight)
        monkeypatch.setattr(run_eval, "run_evaluation", boom)
        return captured

    def test_explicit_preflight_timeout_overrides_both(self, tmp_path, monkeypatch):
        captured = self._capture_and_stop(monkeypatch)
        path = tmp_path / "golden.jsonl"
        _write_golden_set(path, [_golden("p", reviewed=True, expected_sender_type="person")])
        args = _full_args(golden_set=str(path), local_only=True, preflight_timeout=99.0)

        with pytest.raises(RuntimeError, match="reached evaluation"):
            asyncio.run(main(args))
        assert captured["local_timeout"] == 99.0
        assert captured["cloud_timeout"] == 99.0

    def test_preflight_timeouts_default_to_per_endpoint_request_timeouts(self, tmp_path, monkeypatch):
        captured = self._capture_and_stop(monkeypatch)
        path = tmp_path / "golden.jsonl"
        _write_golden_set(path, [_golden("p", reviewed=True, expected_sender_type="person")])
        args = _full_args(golden_set=str(path), local_only=True, preflight_timeout=None)

        with pytest.raises(RuntimeError, match="reached evaluation"):
            asyncio.run(main(args))
        cfg = load_config()
        # Each endpoint defaults to its own request timeout; local is well over 10s
        # so a cold on-demand model load isn't read as "unreachable".
        assert captured["local_timeout"] == cfg["llm"]["local"]["timeout"]
        assert captured["cloud_timeout"] == cfg["llm"]["cloud"]["timeout"]
        assert captured["local_timeout"] > 10


# ---------------------------------------------------------------------------
# Assistant field carried onto the result record (issue #78 groundwork)
# ---------------------------------------------------------------------------

class _StubClassifier:
    """Minimal stand-in for EmailClassifier: fixed answers, no network."""

    def __init__(self, sender="person", label="needs_response"):
        from classifier import EmailLabel, SenderType

        self._sender = SenderType(sender)
        self._label = EmailLabel(label)
        self.cloud_llm = object()
        self.local_llm = object()

    async def classify_sender(self, metadata):
        return self._sender, self._sender.value.upper(), ""

    async def classify_email(self, metadata, transcript, sender_type, vip=False):
        return self._label, self._label.value.upper(), ""

    def _is_vip(self, metadata):
        return False


class TestEvaluateSingleAssistantField:
    def _run(self, golden, stages="full"):
        return asyncio.run(
            run_eval.evaluate_single(golden, _StubClassifier(), stages, DEFAULT_MAX_THREAD_CHARS)
        )[0]

    def test_expected_assistant_is_copied_onto_the_result(self):
        result = self._run(
            _golden("t1", expected_label="needs_response", expected_assistant=True)
        )
        assert result.expected_assistant is True

    def test_unannotated_thread_leaves_the_field_none(self):
        result = self._run(_golden("t2", expected_label="needs_response"))
        assert result.expected_assistant is None

    def test_nothing_predicts_the_assistant_field_yet(self):
        # This slice is plumbing only: no model is asked the question, so the
        # prediction and its correctness stay unset on every path.
        for stages in ("full", "stage1_only", "stage2_only"):
            result = self._run(
                _golden("t3", expected_label="needs_response", expected_assistant=False),
                stages=stages,
            )
            assert result.predicted_assistant is None, stages
            assert result.assistant_correct is None, stages
            assert result.expected_assistant is False, stages
