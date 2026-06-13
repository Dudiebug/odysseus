# Troubleshooting

## Odysseus Doctor

Odysseus Doctor is a read-only, CLI-first diagnostic helper for install and deployment issues. It checks the local Python runtime, repository markers, selected safe configuration values, filesystem paths from `src.constants`, optional Docker status, optional HTTP reachability for `/api/health` and `/api/version`, and the SQLite database in read-only mode when present.

Run it manually from the repository root:

```bash
python scripts/odysseus_doctor.py
```

Run it inside an already-running Docker service:

```bash
docker compose exec odysseus python scripts/odysseus_doctor.py
```

If the app container is not already running, run a temporary container:

```bash
docker compose run --rm odysseus python scripts/odysseus_doctor.py
```

Produce JSON for automation or issue templates:

```bash
python scripts/odysseus_doctor.py --json
```

The plain-text output is intended to be copy/paste friendly for GitHub issues. It should be safe to paste because the doctor redacts secret-looking values and does not print private document contents, raw private logs, API keys, passwords, or tokens. Review output before posting it publicly, especially if your environment uses unusual variable names.

The script is read-only: it does not repair configuration, write config files, create test files, mutate Docker containers, start or stop services, or change databases. Future UI surfaces can reuse the same reusable check logic from `src/diagnostics/doctor.py`, but this first version is intentionally CLI-first.
