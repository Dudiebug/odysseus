"""Read-only install and deployment diagnostics for Odysseus."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

STATUSES = {"PASS", "WARN", "FAIL", "INFO"}
SECRET_RE = re.compile(r"(?i)(token|secret|password|passwd|api[_-]?key|auth|credential|bearer)")
URI_PASSWORD_RE = re.compile(r"(://[^:/@\s]+:)([^@/\s]+)(@)")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)


@dataclass(frozen=True)
class DoctorCheck:
    status: str
    name: str
    detail: str
    hint: str | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"invalid doctor status: {self.status}")

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def sanitize_value(value: object, *, key: str | None = None) -> str:
    """Return a copy/paste-safe diagnostic value without secrets."""
    if value is None:
        return "<unset>"
    text = str(value)
    if PRIVATE_KEY_RE.search(text):
        return "<redacted>"
    if key and SECRET_RE.search(key):
        return "<redacted>" if text else "<unset>"
    text = URI_PASSWORD_RE.sub(r"\1<redacted>\3", text)
    text = re.sub(r"(?i)(password|passwd|token|secret|api[_-]?key)=([^&\s]+)", r"\1=<redacted>", text)
    return text


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return values
    except OSError:
        return values
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _env_value(name: str, dotenv: Mapping[str, str], environ: Mapping[str, str]) -> str | None:
    return environ.get(name, dotenv.get(name))


def _safe_call(name: str, func: Callable[[], Iterable[DoctorCheck]]) -> list[DoctorCheck]:
    try:
        return list(func())
    except Exception as exc:  # defensive: doctor must keep reporting
        return [DoctorCheck("WARN", name, f"Check could not complete: {type(exc).__name__}", "Review the local installation and rerun doctor.")]


def check_python_version(version_info: Sequence[int] | None = None) -> DoctorCheck:
    vi = tuple(version_info or sys.version_info[:3])
    detail = ".".join(str(p) for p in vi[:3])
    if vi >= (3, 11):
        return DoctorCheck("PASS", "Python version", detail)
    return DoctorCheck("FAIL", "Python version", detail, "Use Python 3.11 or newer.")


def _repo_checks(root: Path, git_runner: Callable[..., subprocess.CompletedProcess[str]], timeout: float) -> list[DoctorCheck]:
    checks = [DoctorCheck("INFO", "Platform", platform.platform())]
    markers = [root / "app.py", root / "src" / "constants.py", root / "CONTRIBUTING.md"]
    if all(p.exists() for p in markers):
        checks.append(DoctorCheck("PASS", "Repository", f"Odysseus repo detected at {root}"))
    else:
        checks.append(DoctorCheck("WARN", "Repository", f"Expected Odysseus repo markers were not all found at {root}", "Run from the Odysseus repository root."))
    if shutil.which("git"):
        def git(args: list[str]) -> str:
            cp = git_runner(["git", *args], cwd=str(root), text=True, capture_output=True, timeout=timeout, check=False)
            return (cp.stdout or cp.stderr).strip()
        branch = git(["rev-parse", "--abbrev-ref", "HEAD"])
        commit = git(["rev-parse", "--short", "HEAD"])
        dirty = git(["status", "--porcelain"])
        if branch or commit:
            checks.append(DoctorCheck("INFO", "Git", f"branch={sanitize_value(branch)} commit={sanitize_value(commit)} dirty={'yes' if dirty else 'no'}"))
    else:
        checks.append(DoctorCheck("INFO", "Git", "git command not found"))
    return checks


def _config_checks(root: Path, dotenv: Mapping[str, str], environ: Mapping[str, str]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    env_path = root / ".env"
    checks.append(DoctorCheck("PASS" if env_path.exists() else "WARN", ".env", "present" if env_path.exists() else "missing", "Copy .env.example to .env if this is a local install." if not env_path.exists() else None))
    names = ["APP_BIND", "APP_PORT", "AUTH_ENABLED", "DATABASE_URL", "ODYSSEUS_DATA_DIR", "LOCALHOST_BYPASS"]
    for name in names:
        checks.append(DoctorCheck("INFO", f"Config {name}", sanitize_value(_env_value(name, dotenv, environ), key=name)))
    auth = (_env_value("AUTH_ENABLED", dotenv, environ) or "").lower() == "false"
    bind_all = (_env_value("APP_BIND", dotenv, environ) or "") == "0.0.0.0"
    bypass = (_env_value("LOCALHOST_BYPASS", dotenv, environ) or "").lower() == "true"
    if auth:
        checks.append(DoctorCheck("WARN", "AUTH_ENABLED", "AUTH_ENABLED=false", "Enable authentication before exposing Odysseus."))
    if bind_all:
        checks.append(DoctorCheck("WARN", "APP_BIND", "APP_BIND=0.0.0.0", "Only bind publicly behind appropriate network controls."))
    if auth and bind_all:
        checks.append(DoctorCheck("WARN", "Public unauthenticated binding", "AUTH_ENABLED=false while APP_BIND=0.0.0.0", "Do not expose unauthenticated Odysseus to a network."))
    if bypass:
        checks.append(DoctorCheck("WARN", "LOCALHOST_BYPASS", "LOCALHOST_BYPASS=true", "Disable localhost bypass unless you understand the trust boundary."))
    return checks


def _path_checks() -> list[DoctorCheck]:
    from src import constants
    names = ["DATA_DIR", "APP_DB", "SESSIONS_FILE", "AUTH_FILE", "SETTINGS_FILE", "PRESETS_FILE", "UPLOAD_DIR", "PERSONAL_DIR", "MEMORY_FILE", "USER_PREFS_FILE", "CHROMA_DIR"]
    checks: list[DoctorCheck] = []
    for name in names:
        path = Path(getattr(constants, name))
        exists = path.exists()
        status = "PASS" if exists else "INFO"
        checks.append(DoctorCheck(status, f"Path {name}", f"{sanitize_value(path)} ({'exists' if exists else 'missing'})", "May be created by Odysseus on first run." if not exists else None))
        target = path if path.is_dir() else path.parent
        if target.exists():
            writable = os.access(target, os.W_OK)
            checks.append(DoctorCheck("PASS" if writable else "FAIL", f"Writable {name}", f"{sanitize_value(target)} writable={writable}", "Fix filesystem permissions for the Odysseus process." if not writable else None))
    return checks


def _network_checks(timeout: float) -> list[DoctorCheck]:
    from src.constants import internal_api_base
    base = internal_api_base()
    checks = []
    for endpoint in ("/api/health", "/api/version"):
        url = f"{base}{endpoint}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec - internal diagnostic URL
                checks.append(DoctorCheck("PASS", f"HTTP {endpoint}", f"status={resp.status}"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            checks.append(DoctorCheck("WARN", f"HTTP {endpoint}", f"not reachable: {type(exc).__name__}", "Start Odysseus or rerun with --no-network."))
    return checks


def _docker_checks(timeout: float, runner: Callable[..., subprocess.CompletedProcess[str]]) -> list[DoctorCheck]:
    if not shutil.which("docker"):
        return [DoctorCheck("INFO", "Docker", "docker command not found")]
    checks = [DoctorCheck("PASS", "Docker", "docker command found")]
    cp = runner(["docker", "compose", "version"], text=True, capture_output=True, timeout=timeout, check=False)
    if cp.returncode != 0:
        checks.append(DoctorCheck("WARN", "Docker Compose", "docker compose is not available", "Install Docker Compose or rerun with --no-docker."))
        return checks
    checks.append(DoctorCheck("PASS", "Docker Compose", sanitize_value((cp.stdout or cp.stderr).strip())))
    ps = runner(["docker", "compose", "ps"], text=True, capture_output=True, timeout=timeout, check=False)
    checks.append(DoctorCheck("PASS" if ps.returncode == 0 else "WARN", "Docker Compose ps", "completed" if ps.returncode == 0 else "docker compose ps failed"))
    return checks


def _database_checks() -> list[DoctorCheck]:
    from src.constants import APP_DB
    db = Path(APP_DB)
    if not db.exists():
        return [DoctorCheck("INFO", "SQLite database", f"{sanitize_value(db)} missing", "It may be created on first run.")]
    try:
        uri = f"file:{db.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            conn.execute("PRAGMA schema_version").fetchone()
        finally:
            conn.close()
        return [DoctorCheck("PASS", "SQLite database", "opened read-only successfully")]
    except sqlite3.Error as exc:
        return [DoctorCheck("WARN", "SQLite database", f"read-only open failed: {type(exc).__name__}", "Check database file permissions and integrity.")]


def run_doctor_checks(*, root: Path | None = None, include_network: bool = True, include_docker: bool = True, timeout: float = 3.0, environ: Mapping[str, str] | None = None, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[DoctorCheck]:
    root = Path(root or os.getcwd()).resolve()
    environ = environ or os.environ
    dotenv = load_env_file(root / ".env")
    checks: list[DoctorCheck] = [check_python_version()]
    groups = [
        ("Repository", lambda: _repo_checks(root, runner, timeout)),
        ("Configuration", lambda: _config_checks(root, dotenv, environ)),
        ("Paths", _path_checks),
        ("Database", _database_checks),
    ]
    if include_network:
        groups.append(("Network", lambda: _network_checks(timeout)))
    else:
        checks.append(DoctorCheck("INFO", "Network checks", "skipped by request"))
    if include_docker:
        groups.append(("Docker", lambda: _docker_checks(timeout, runner)))
    else:
        checks.append(DoctorCheck("INFO", "Docker checks", "skipped by request"))
    for name, func in groups:
        checks.extend(_safe_call(name, func))
    return checks


def format_text_report(checks: Iterable[DoctorCheck]) -> str:
    lines = []
    for check in checks:
        line = f"{check.status:<5} {check.name}: {sanitize_value(check.detail)}"
        if check.hint:
            line += f" Hint: {sanitize_value(check.hint)}"
        lines.append(line)
    return "\n".join(lines)


def format_json_report(checks: Iterable[DoctorCheck]) -> str:
    return json.dumps({"checks": [c.to_dict() for c in checks]}, indent=2, sort_keys=True)
