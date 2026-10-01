"""The real stdio server: stdout carries JSON-RPC and nothing else (K1).

These start the server as a subprocess against the temp vault from conftest (the
SIFT_* variables are inherited), with the model warm-up and the index sync switched
off so nothing loads a model or touches the network. They never call search_memory,
which would need a real embedder in the child process.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading

import pytest

TIMEOUT_S = 60

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"},
    },
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def _call(msg_id: int, name: str, arguments: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def _env() -> dict[str, str]:
    env = dict(os.environ)  # carries the monkeypatched SIFT_VAULT_PATH / SIFT_DB_PATH
    env.update(
        SIFT_MCP_WARMUP="false",
        SIFT_MCP_AUTO_SYNC="false",
        FASTMCP_CHECK_FOR_UPDATES="off",
        HF_HUB_OFFLINE="1",
        PYTHONIOENCODING="utf-8",
    )
    return env


def _write_vault(vault) -> None:
    (vault / "finding").mkdir(parents=True, exist_ok=True)
    (vault / "finding" / "Good note.md").write_text(
        "---\nid: good-note\ntype: finding\ntitle: Good note\n---\n\nbody\n", encoding="utf-8"
    )
    (vault / "report").mkdir(exist_ok=True)
    (vault / "report" / "empty.md").write_bytes(b"")  # what a crashed write used to leave
    (vault / "START HERE.md").write_text("No frontmatter here.\n", encoding="utf-8")
    (vault / "finding" / "Broken.md").write_text(
        "---\nid: broken\ntype: [not, a, type]\n---\n\nbody\n", encoding="utf-8"
    )


def _session(argv: list[str], calls: list[dict]) -> tuple[list[bytes], bytes, dict[int, dict]]:
    """Run one server session: send each message, waiting for the reply to each
    request, then close stdin and collect everything the server wrote."""
    proc = subprocess.Popen(
        [sys.executable, *argv],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_env(),
    )
    lines: queue.Queue[bytes | None] = queue.Queue()

    def pump() -> None:
        for raw in proc.stdout:
            lines.put(raw)
        lines.put(None)

    stderr: list[bytes] = []
    threading.Thread(target=pump, daemon=True).start()
    err_reader = threading.Thread(target=lambda: stderr.append(proc.stderr.read()), daemon=True)
    err_reader.start()

    seen: list[bytes] = []
    responses: dict[int, dict] = {}

    def take(raw: bytes | None) -> None:
        if raw is None:
            return
        seen.append(raw)
        if raw.strip().startswith(b"{"):
            msg = json.loads(raw)
            if "id" in msg:
                responses[msg["id"]] = msg

    try:
        for msg in calls:
            proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
            proc.stdin.flush()
            if "id" not in msg:
                continue
            while msg["id"] not in responses:
                try:
                    raw = lines.get(timeout=TIMEOUT_S)
                except queue.Empty:
                    pytest.fail(f"no response to {msg['id']}; stderr: {b''.join(stderr)[-2000:]!r}")
                if raw is None:
                    pytest.fail(f"server exited early; stderr: {b''.join(stderr)[-2000:]!r}")
                take(raw)
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    while True:  # what was written while shutting down
        try:
            raw = lines.get(timeout=10)
        except queue.Empty:
            break
        if raw is None:
            break
        take(raw)
    err_reader.join(timeout=10)
    return seen, b"".join(stderr), responses


def _non_json(seen: list[bytes]) -> list[bytes]:
    bad = []
    for raw in seen:
        if not raw.strip():
            continue
        try:
            json.loads(raw)
        except ValueError:
            bad.append(raw)
    return bad


def test_stdout_is_pure_json_rpc(vault_path):
    _write_vault(vault_path)
    seen, _err, responses = _session(
        ["-m", "sift.mcp_server"],
        [
            INITIALIZE,
            INITIALIZED,
            _call(2, "list_notes", {}),
            _call(3, "stats", {}),
            _call(4, "get_note", {"slug": "nothing-here"}),
        ],
    )
    # Every byte on stdout is a JSON-RPC frame: the skipped files, the warm-up thread
    # and shutdown left nothing behind on the wire.
    assert _non_json(seen) == []

    listed = responses[2]["result"]
    assert listed.get("isError") in (None, False)
    payload = listed.get("structuredContent") or json.loads(listed["content"][0]["text"])
    assert payload["count"] == 1
    assert payload["notes"][0]["note_id"] == "good-note"
    st = responses[3]["result"]
    st = st.get("structuredContent") or json.loads(st["content"][0]["text"])
    assert st["total_notes"] == 1 and st["skipped_notes"] == 3
    assert responses[4]["result"].get("isError") is True


BANNER_MARK = b"gofastmcp.com"  # in FastMCP's startup banner (fastmcp.utilities.cli)


def test_sift_mcp_serves_pure_json_rpc_without_a_banner(vault_path):
    """`sift mcp` - the command .mcp.json runs. The CLI's own setup (its logging, the
    import-time stdout redirect, the FastMCP defaults) must leave stdout pure
    JSON-RPC through shutdown, and skip FastMCP's banner, which is also what triggers
    its synchronous PyPI version check before serving."""
    _write_vault(vault_path)
    seen, err, responses = _session(
        ["-m", "sift.cli", "mcp"],
        [INITIALIZE, INITIALIZED, _call(2, "list_notes", {}), _call(3, "stats", {})],
    )
    assert _non_json(seen) == []
    assert responses[1]["result"]["serverInfo"]["name"] == "sift"
    listed = responses[2]["result"]
    payload = listed.get("structuredContent") or json.loads(listed["content"][0]["text"])
    assert payload["count"] == 1
    assert BANNER_MARK not in err


def test_the_banner_check_can_fail():
    """Positive control for the banner assertion above: a plain `mcp.run()` prints it.
    (FASTMCP_CHECK_FOR_UPDATES=off from _env keeps this run offline.)"""
    seen, err, responses = _session(
        ["-c", "import sift.mcp_server as m; m.mcp.run(transport='stdio')"], [INITIALIZE]
    )
    assert 1 in responses
    assert BANNER_MARK in err


STRAY_PRINT = (
    "import sift.mcp_server as m; real = m._catalog; "
    "m._catalog = lambda *a, **k: (print('STRAY-PRINT'), real(*a, **k))[1]; "
)


def test_a_stray_print_never_reaches_the_wire(vault_path):
    """`main` line-buffers stdout before serving, so a print() that slips into a
    tool flushes at once into the fd stdio diverted to stderr. Without it the
    buffered line was written onto the wire at shutdown (verified: `mcp.run()`
    alone leaves 'STRAY-PRINT' on stdout)."""
    _write_vault(vault_path)
    seen, err, responses = _session(
        ["-c", STRAY_PRINT + "m.main()"],
        [INITIALIZE, INITIALIZED, _call(2, "list_notes", {})],
    )
    assert 2 in responses
    assert _non_json(seen) == []
    assert b"STRAY-PRINT" in err  # positive control: the print happened, on stderr


def test_import_is_light():
    """Importing the server must not pull in LanceDB, torch or the embedders: they
    load in the tools (and on the warm-up thread), not before `initialize`."""
    code = (
        "import json, sys; import sift.mcp_server; "
        "print(json.dumps(sorted(m for m in ('lancedb', 'torch', 'sentence_transformers', "
        "'fastembed', 'sift.pipeline', 'sift.index.store') if m in sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=_env(),
        timeout=TIMEOUT_S,
        check=True,
    )
    assert json.loads(out.stdout.decode().strip().splitlines()[-1]) == []
