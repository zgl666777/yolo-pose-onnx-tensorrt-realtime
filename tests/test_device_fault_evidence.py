import importlib.util
from pathlib import Path


def load_tool(name):
    source = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rtsp_evidence_redacts_credentials_and_query():
    tool = load_tool("device_fault_evidence")
    assert tool.redact_url("rtsp://operator:secret@192.0.2.4:8554/live?token=secret") == "rtsp://192.0.2.4:8554/live"
    assert tool.redact_url("rtsp://operator:secret@[::1]:8554/live") == "rtsp://[::1]:8554/live"


def test_invalid_rtsp_port_is_evidence_not_unhandled_exception():
    tool = load_tool("device_fault_evidence")
    assert tool.probe_tcp("rtsp://localhost:not-a-port/live")["status"] == "invalid"


def test_rtsp_recovery_requires_actual_frame_delivery():
    tool = load_tool("rtsp_fault_test")
    summary = dict(status="completed", processed=20, reconnects=2,
                   capture_thread_alive=False, capture_worker_alive=False, capture_error=None)
    health = [dict(event="source_opened", generation=g, elapsed_seconds=g * 10) for g in range(3)]
    faults = [dict(event=name, elapsed_seconds=elapsed) for name, elapsed in
              (("publisher_terminated", 5), ("publisher_restarted", 8),
               ("server_terminated", 15), ("server_and_publisher_restarted", 18))]
    missing_recovery = [dict(generation=0, timestamp=1)]
    assert tool.analyze_recovery(summary, health, faults, 0, missing_recovery)["status"] == "failed"
    recovered = [dict(generation=g, timestamp=g * 10 + 1) for g in range(3)]
    assert tool.analyze_recovery(summary, health, faults, 0, recovered)["status"] == "passed"
    opens_before_fault = [dict(event="source_opened", generation=g, elapsed_seconds=0) for g in range(3)]
    assert tool.analyze_recovery(summary, opens_before_fault, faults, 0, recovered)["status"] == "failed"


def test_rtsp_failed_pipeline_is_never_accepted():
    tool = load_tool("rtsp_fault_test")
    summary = dict(status="failed", processed=20, reconnects=2,
                   capture_thread_alive=False, capture_worker_alive=False, capture_error=None)
    health = [dict(event="source_opened", generation=g) for g in range(3)]
    frames = [dict(generation=g, timestamp=g * 10 + 1) for g in range(3)]
    assert tool.analyze_recovery(summary, health, [dict(event="server_terminated")], 0, frames)["status"] == "failed"


def test_manual_fault_requires_operator_actions_and_measured_recovery():
    tool = load_tool("manual_fault_test")
    summary = dict(status="completed", capture_thread_alive=False, capture_worker_alive=False, capture_error=None)
    health = [dict(event="source_read_failed", elapsed_seconds=2),
              dict(event="source_opened", elapsed_seconds=4, generation=1)]
    frames = [dict(generation=0), dict(generation=1)]
    actions = [dict(action="disconnect_completed", elapsed_seconds=1),
               dict(action="reconnect_completed", elapsed_seconds=3)]
    assert tool.summarize_recovery(summary, health, frames, actions)["status"] == "passed_operator_attested"
    assert tool.summarize_recovery(summary, health, frames, actions[:1])["status"] == "failed"
    assert tool.summarize_recovery(summary, [], frames, actions)["status"] == "failed"
