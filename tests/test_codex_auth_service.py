import asyncio
import os

from src.codex_auth import CodexAuthService


def run(coro):
    return asyncio.run(coro)


def test_status_reports_missing_cli_without_local_diagnostics():
    svc = CodexAuthService(codex_bin="definitely-not-codex", enabled=True)
    out = run(svc.status())
    allowed_keys = {
        "enabled",
        "method",
        "sdk_available",
        "sdk_status",
        "codex_cli_available",
        "stores_credentials",
        "codex_authenticated",
        "device_login_active",
        "cli_found",
        "cli_executable",
        "status",
        "message",
        "authenticated",
        "auth_mode",
        "started_at",
        "updated_at",
        "error_code",
        "process_running",
    }
    assert out["status"] == "cli_missing"
    assert out["error_code"] == "cli_missing"
    assert out["codex_cli_available"] is False
    assert out["cli_found"] is False
    assert out["cli_executable"] is False
    assert set(out) <= allowed_keys


def test_status_parses_chatgpt_login(monkeypatch):
    svc = CodexAuthService(enabled=True)
    monkeypatch.setattr(svc, "_bin_path", lambda: "/usr/bin/codex")

    async def fake_run(args, timeout=15.0):
        return 0, "Logged in using ChatGPT\n"

    monkeypatch.setattr(svc, "_run_command", fake_run)
    out = run(svc.status())
    assert out["status"] == "authenticated"
    assert out["authenticated"] is True
    assert out["codex_authenticated"] is True
    assert out["auth_mode"] == "ChatGPT"


def test_status_does_not_echo_unrecognized_cli_output(monkeypatch):
    svc = CodexAuthService(enabled=True)
    monkeypatch.setattr(svc, "_bin_path", lambda: "/usr/bin/codex")

    async def fake_run(args, timeout=15.0):
        return 0, "access_token=secret refresh_token=secret"

    monkeypatch.setattr(svc, "_run_command", fake_run)
    out = run(svc.status())
    assert out["status"] == "not_authenticated"
    assert "secret" not in out["message"]
    assert "token" not in out["message"].lower()


def test_cli_credentials_delegated_without_auth_file_access(monkeypatch, tmp_path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"access_token":"secret"}')
    real_open = open

    def guarded_open(file, *args, **kwargs):
        if os.fspath(file) == os.fspath(auth_file):
            raise AssertionError("Odysseus must not read or write Codex auth.json")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)
    svc = CodexAuthService(codex_bin="definitely-not-codex", enabled=True)
    out = run(svc.status())
    assert out["status"] == "cli_missing"
    assert "secret" not in str(out)
    assert str(tmp_path) not in str(out)


def test_disabled_state():
    svc = CodexAuthService(enabled=False)
    out = run(svc.start())
    assert out["status"] == "disabled"
    assert out["error_code"] == "disabled"


class _FakeStdout:
    def __init__(self, lines):
        self._lines = [line.encode() for line in lines]

    async def readline(self):
        await asyncio.sleep(0)
        if self._lines:
            return self._lines.pop(0)
        return b""


class _FakeProcess:
    def __init__(self, lines, returncode=0):
        self.stdout = _FakeStdout(lines)
        self.returncode = None
        self._final_returncode = returncode
        self.terminated = False

    async def wait(self):
        self.returncode = self._final_returncode
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.terminated = True
        self.returncode = -9


class _SlowStdout:
    async def readline(self):
        await asyncio.sleep(2)
        return b""


def test_start_device_flow_returns_device_code(monkeypatch):
    svc = CodexAuthService(enabled=True, code_timeout_seconds=1)
    fake_proc = _FakeProcess([
        "Open https://auth.openai.com/codex/device\n",
        "Enter code CODE-12345\n",
        "Successfully logged in\n",
    ])
    monkeypatch.setattr(svc, "_bin_path", lambda: "/usr/bin/codex")

    async def fake_run(args, timeout=15.0):
        return 1, "Not logged in\n"

    async def fake_exec(*args, **kwargs):
        return fake_proc

    monkeypatch.setattr(svc, "_run_command", fake_run)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def scenario():
        first = await svc.start()
        assert first["status"] in {"pending", "succeeded"}
        if first["status"] == "pending":
            assert first["verification_url"] == "https://auth.openai.com/codex/device"
            assert first["user_code"] == "CODE-12345"
        status = await svc.status()
        assert "verification_url" not in status
        assert "user_code" not in status
        await asyncio.sleep(0.05)
        assert svc._state.status == "succeeded"
        assert svc._state.authenticated is True

    run(scenario())


def test_start_device_flow_times_out_waiting_for_code(monkeypatch):
    svc = CodexAuthService(enabled=True, code_timeout_seconds=0)
    fake_proc = _FakeProcess([], returncode=1)
    fake_proc.stdout = _SlowStdout()
    monkeypatch.setattr(svc, "_bin_path", lambda: "/usr/bin/codex")

    async def fake_run(args, timeout=15.0):
        return 1, "Not logged in\n"

    async def fake_exec(*args, **kwargs):
        return fake_proc

    monkeypatch.setattr(svc, "_run_command", fake_run)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def scenario():
        await svc.start()
        await asyncio.sleep(1.1)
        assert svc._state.status == "failed"
        assert svc._state.error_code == "device_code_unavailable"
        assert fake_proc.terminated is True

    run(scenario())


def test_start_device_flow_reports_cli_failure(monkeypatch):
    svc = CodexAuthService(enabled=True)
    fake_proc = _FakeProcess([
        "Error logging in with device code: device code login is not enabled for this Codex server\n",
    ], returncode=1)
    monkeypatch.setattr(svc, "_bin_path", lambda: "/usr/bin/codex")

    async def fake_run(args, timeout=15.0):
        return 1, "Not logged in\n"

    async def fake_exec(*args, **kwargs):
        return fake_proc

    monkeypatch.setattr(svc, "_run_command", fake_run)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def scenario():
        await svc.start()
        await asyncio.sleep(0.05)
        assert svc._state.status == "failed"
        assert svc._state.error_code == "device_auth_disabled"

    run(scenario())


def test_logout_runs_codex_logout(monkeypatch):
    svc = CodexAuthService(enabled=True)
    calls = []
    monkeypatch.setattr(svc, "_bin_path", lambda: "/usr/bin/codex")

    async def fake_run(args, timeout=15.0):
        calls.append(args)
        return 0, "Successfully logged out\n"

    monkeypatch.setattr(svc, "_run_command", fake_run)
    out = run(svc.logout())
    assert out["status"] == "logged_out"
    assert out["message"] == "Codex credentials removed"
    assert ["logout"] in calls


def test_status_after_logout_rechecks_codex_login_status(monkeypatch):
    svc = CodexAuthService(enabled=True)
    calls = []
    monkeypatch.setattr(svc, "_bin_path", lambda: "/usr/bin/codex")

    async def fake_run(args, timeout=15.0):
        calls.append(args)
        if args == ["logout"]:
            return 0, "Successfully logged out\n"
        return 1, "Not logged in\n"

    monkeypatch.setattr(svc, "_run_command", fake_run)
    out = run(svc.logout())
    assert out["status"] == "logged_out"

    status = run(svc.status())
    assert status["status"] == "not_authenticated"
    assert status["message"] == "Codex CLI ready. Not signed in."
    assert calls == [["logout"], ["login", "status"]]


def test_logout_failure_does_not_echo_cli_output(monkeypatch):
    svc = CodexAuthService(enabled=True)
    monkeypatch.setattr(svc, "_bin_path", lambda: "/usr/bin/codex")

    async def fake_run(args, timeout=15.0):
        return 1, "refresh_token=secret"

    monkeypatch.setattr(svc, "_run_command", fake_run)
    out = run(svc.logout())
    assert out["status"] == "failed"
    assert out["message"] == "Codex logout failed"
    assert "secret" not in str(out)
