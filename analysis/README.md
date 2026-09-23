# Dependency Workflow

This analysis workspace now uses:

- `requirements.txt`: human-edited top-level requirements
- `requirements.lock.txt`: pinned lockfile generated from `requirements.txt`

Refresh the lockfile:

```bash
uv pip compile requirements.txt -o requirements.lock.txt
```

Sync the local environment to the lockfile:

```bash
uv pip sync --python .venv/bin/python requirements.lock.txt
```

Install the audit tool in the local environment when needed:

```bash
uv pip install --python .venv/bin/python pip-audit
```

Audit the pinned lockfile:

```bash
.venv/bin/pip-audit -r requirements.lock.txt --no-deps --disable-pip
```

Audit the installed environment:

```bash
.venv/bin/pip-audit
```

`requirements.txt` uses loose ranges, so `requirements.lock.txt` is the actual audited artifact.
