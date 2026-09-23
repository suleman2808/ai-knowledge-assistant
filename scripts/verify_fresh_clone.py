"""Verify that the project works from a genuine fresh clone.

    python -m scripts.verify_fresh_clone
    python -m scripts.verify_fresh_clone --keep      # leave the clone in place

Clones this repository into a temporary directory, builds a new
virtualenv, installs only what `requirements.txt` declares, and then
follows the README's quickstart in order: smoke test, ingest, tests, a
CLI query, the server, and an HTTP request whose answer must come back
grounded. It also checks the clone for leaked secrets.

The point is to catch the things that only break for someone else: a
file that exists on the developer's machine but was never committed, a
dependency installed by hand months ago, a setup step that lives only in
somebody's memory. Running the app in the directory you built it in
proves none of that.

The one thing carried across is `.env`, because an API key cannot be
invented. Everything else must come from the repository.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from app.config import PROJECT_ROOT

PORT = 8137
# Files that must never appear in a clone.
SECRETS = (".env", "credentials.json", "token.json", "data/analytics.db", "data/chroma")


class Report:
    """Collects pass/fail results so one failure does not stop the run."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{f'  — {detail}' if detail else ''}")
        if not ok:
            self.failures.append(name)
        return ok

    def heading(self, name: str) -> None:
        print(f"\n{'=' * 68}\n{name}\n{'=' * 68}")


def run(command: list[str], cwd: Path, *, timeout: int = 900) -> tuple[int, str]:
    """Run a command, returning its exit code and combined output."""
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def python_in(venv: Path) -> Path:
    """Path to the interpreter inside a virtualenv, on either platform."""
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def wait_for_health(url: str, *, attempts: int = 40) -> dict | None:
    for _ in range(attempts):
        time.sleep(1.5)
        try:
            with urllib.request.urlopen(url, timeout=4) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            continue
    return None


def post_chat(url: str, message: str) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps({"message": message}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Do not delete the clone.")
    args = parser.parse_args()

    report = Report()
    workdir = Path(tempfile.mkdtemp(prefix="freshclone-"))
    clone = workdir / "repo"

    report.heading(f"Cloning into {clone}")
    code, output = run(["git", "clone", "--quiet", str(PROJECT_ROOT), str(clone)], workdir)
    if code != 0:
        print(output)
        return 1
    print("  cloned")

    report.heading("No secrets in the clone")
    for secret in SECRETS:
        report.check(f"absent: {secret}", not (clone / secret).exists())

    report.heading("Install")
    venv = clone / ".venv"
    code, output = run([sys.executable, "-m", "venv", str(venv)], clone)
    if not report.check("create virtualenv", code == 0, output[-200:] if code else ""):
        return 1

    python = python_in(venv)
    code, output = run(
        [str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
         "-r", "requirements.txt"],
        clone,
    )
    if not report.check("pip install -r requirements.txt", code == 0, output[-300:] if code else ""):
        return 1

    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        print("\n  No .env to copy — create one first; the rest needs an API key.")
        return 1
    shutil.copy(env_file, clone / ".env")
    print("  copied .env (the API key cannot come from the repository)")

    report.heading("The README's quickstart, in order")
    for name, command in [
        ("smoke test", ["-m", "scripts.smoke_test"]),
        ("ingest", ["-m", "scripts.ingest", "--quiet"]),
        ("pytest", ["-m", "pytest", "tests/", "-q"]),
        ("search", ["-m", "scripts.search", "do you take cigna"]),
        ("chat", ["-m", "scripts.chat", "what time do you close on Friday"]),
    ]:
        code, output = run([str(python), *command], clone)
        report.check(name, code == 0, output.strip().splitlines()[-1] if code else "")

    report.heading("Server and HTTP endpoints")
    server = subprocess.Popen(
        [str(python), "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
        cwd=clone, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        health = wait_for_health(f"http://127.0.0.1:{PORT}/api/health")
        if not report.check("server starts and reports health", health is not None):
            return 1
        assert health is not None

        print(f"        status={health['status']}  "
              f"chunks={health['checks']['knowledge_base'].get('chunks')}  "
              f"embeddings={health['checks']['embeddings'].get('backend')}  "
              f"calendar={health['checks']['calendar'].get('backend')}")

        try:
            reply = post_chat(f"http://127.0.0.1:{PORT}/api/chat",
                              "how much is a root canal on a molar")
            report.check("chat answers", bool(reply.get("answer")))
            # The whole point of the project: the answer must come from the
            # documents, not from the model's own knowledge.
            report.check("answer is grounded", reply.get("grounded") is True)
            report.check("answer cites sources", len(reply.get("sources", [])) > 0)
            print(f"        {reply.get('answer', '')[:100]}")
        except Exception as exc:
            report.check("chat answers", False, str(exc))

        for path in ("/", "/analytics", "/docs", "/static/app.js"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=15) as r:
                    report.check(f"GET {path}", r.status == 200, str(r.status))
            except Exception as exc:
                report.check(f"GET {path}", False, str(exc))
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    report.heading("Result")
    if report.failures:
        print(f"  {len(report.failures)} check(s) failed:")
        for failure in report.failures:
            print(f"    {failure}")
    else:
        print("  ALL CHECKS PASSED — the project works from a fresh clone.")

    if args.keep:
        print(f"\n  Clone kept at {clone}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)

    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
