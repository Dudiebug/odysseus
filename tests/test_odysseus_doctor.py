import json
from pathlib import Path

from src.diagnostics import doctor
from src.diagnostics.doctor import DoctorCheck


def test_text_result_formatting():
    text = doctor.format_text_report([DoctorCheck("PASS", "Python version", "3.11.0")])
    assert text == "PASS  Python version: 3.11.0"


def test_json_result_formatting():
    data = json.loads(doctor.format_json_report([DoctorCheck("WARN", "Thing", "detail", "hint")]))
    assert data == {"checks": [{"status": "WARN", "name": "Thing", "detail": "detail", "hint": "hint"}]}


def test_secret_redaction_sanitization():
    assert doctor.sanitize_value("abc", key="OPENAI_API_KEY") == "<redacted>"
    assert "pw" not in doctor.sanitize_value("postgres://user:pw@db/app")
    assert "<redacted>" in doctor.sanitize_value("url?token=secret")


def test_unsafe_deployment_flag_logic(tmp_path):
    checks = doctor.run_doctor_checks(
        root=tmp_path,
        include_network=False,
        include_docker=False,
        environ={"AUTH_ENABLED": "false", "APP_BIND": "0.0.0.0", "LOCALHOST_BYPASS": "true"},
    )
    names = {c.name: c for c in checks}
    assert names["AUTH_ENABLED"].status == "WARN"
    assert names["APP_BIND"].status == "WARN"
    assert names["Public unauthenticated binding"].status == "WARN"
    assert names["LOCALHOST_BYPASS"].status == "WARN"


def test_python_version_check_behavior():
    assert doctor.check_python_version((3, 11, 0)).status == "PASS"
    assert doctor.check_python_version((3, 10, 9)).status == "FAIL"


def test_missing_env_behavior(tmp_path):
    checks = doctor.run_doctor_checks(root=tmp_path, include_network=False, include_docker=False, environ={})
    env_check = next(c for c in checks if c.name == ".env")
    assert env_check.status == "WARN"
    assert "missing" in env_check.detail


def test_path_check_behavior_with_temp_constants(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = data_dir / "app.db"
    db.write_bytes(b"")
    for name in ("DATA_DIR", "UPLOAD_DIR", "PERSONAL_DIR", "CHROMA_DIR"):
        monkeypatch.setattr("src.constants." + name, str(data_dir / name.lower()), raising=False)
        Path(getattr(__import__("src.constants", fromlist=[name]), name)).mkdir(exist_ok=True)
    for name in ("APP_DB", "SESSIONS_FILE", "AUTH_FILE", "SETTINGS_FILE", "PRESETS_FILE", "MEMORY_FILE", "USER_PREFS_FILE"):
        monkeypatch.setattr("src.constants." + name, str(data_dir / f"{name.lower()}.json"), raising=False)
    monkeypatch.setattr("src.constants.APP_DB", str(db), raising=False)
    checks = doctor.run_doctor_checks(root=tmp_path, include_network=False, include_docker=False, environ={})
    assert any(c.name == "Path DATA_DIR" and c.status == "PASS" for c in checks)
    assert any(c.name == "SQLite database" for c in checks)


def test_network_checks_skipped(tmp_path):
    checks = doctor.run_doctor_checks(root=tmp_path, include_network=False, include_docker=False, environ={})
    assert any(c.name == "Network checks" and c.detail == "skipped by request" for c in checks)


def test_docker_checks_degrade_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.shutil, "which", lambda cmd: None if cmd == "docker" else "/usr/bin/git")
    checks = doctor.run_doctor_checks(root=tmp_path, include_network=False, include_docker=True, environ={})
    assert any(c.name == "Docker" and c.status == "INFO" for c in checks)


def test_failing_check_does_not_prevent_full_report(monkeypatch, tmp_path):
    def boom():
        raise RuntimeError("nope")

    monkeypatch.setattr(doctor, "_path_checks", boom)
    checks = doctor.run_doctor_checks(root=tmp_path, include_network=False, include_docker=False, environ={})
    assert any(c.name == "Paths" and c.status == "WARN" for c in checks)
    assert "Python version" in doctor.format_text_report(checks)
