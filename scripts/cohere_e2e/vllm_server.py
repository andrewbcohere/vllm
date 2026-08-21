# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared harness for the Cohere e2e scripts that need a live vLLM server.

The scripts in this directory drive a real ``vllm serve`` process over
HTTP. Rather than making the caller start one by hand, they can point at
an existing ``--base-url`` and fall back to launching (and tearing down)
a server themselves.

Two halves that are meant to be used together:

* :func:`add_server_args` registers the CLI flags this module consumes.
* :func:`ensure_server` reads the resulting namespace and returns a
  :class:`ManagedServer` when it had to start one (``None`` when the
  caller-supplied URL was already healthy, so we own nothing).

A typical ``main()`` looks like::

    parser = argparse.ArgumentParser(description=__doc__)
    add_server_args(parser, default_model=DEFAULT_MODEL)
    ...
    args = parser.parse_args()

    managed = ensure_server(args, log_prefix="vllm-cohere-foo-e2e-")
    try:
        ...  # run the tests against args.base_url
    finally:
        release_server(managed, keep=args.keep_server)

Because these are standalone scripts (no ``__init__.py`` here), import
this as a flat sibling module -- ``from vllm_server import ...`` -- which
resolves via the script directory Python puts on ``sys.path``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def server_is_healthy(base_url: str, timeout: float = 5.0) -> bool:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{base_url.rstrip('/')}/health")
        return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


@dataclass
class ManagedServer:
    proc: subprocess.Popen
    log_path: str

    def stop(self, *, timeout: float = 30.0) -> None:
        if self.proc.poll() is not None:
            return
        pgid: int | None = None
        with contextlib.suppress(ProcessLookupError, OSError):
            pgid = os.getpgid(self.proc.pid)
        target = -pgid if pgid is not None else self.proc.pid
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(target, signal.SIGTERM)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(target, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=10)


def build_serve_command(
    *,
    model: str,
    host: str,
    port: int,
    is_reasoning_model: bool,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--tokenizer-mode",
        "cohere",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "cohere2",
        "--reasoning-parser",
        "cohere2",
    ]
    if not is_reasoning_model:
        cmd.append("--no-cohere-is-reasoning-model")
    cmd.extend(extra_args)
    return cmd


def start_server(
    *,
    model: str,
    host: str,
    port: int,
    is_reasoning_model: bool,
    extra_args: list[str],
    log_prefix: str = "vllm-cohere-e2e-",
) -> ManagedServer:
    cmd = build_serve_command(
        model=model,
        host=host,
        port=port,
        is_reasoning_model=is_reasoning_model,
        extra_args=extra_args,
    )
    env = os.environ.copy()
    env["VLLM_ENABLE_COHERE_API"] = "1"

    log_fd, log_path = tempfile.mkstemp(prefix=log_prefix, suffix=".log")
    os.close(log_fd)
    log_file = open(log_path, "w")  # noqa: SIM115 -- lives as long as the server

    print(f"Server not reachable; starting it ourselves:\n  {' '.join(cmd)}")
    print(f"  logs: {log_path}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return ManagedServer(proc=proc, log_path=log_path)


def wait_until_healthy(*, base_url: str, server: ManagedServer, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        exit_code = server.proc.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"vLLM server process exited early (code={exit_code}); "
                f"see {server.log_path} for details."
            )
        if server_is_healthy(base_url):
            print(f"Server is healthy at {base_url}")
            return
        time.sleep(5.0)
    raise TimeoutError(
        f"vLLM server did not become healthy within {timeout:.0f}s; "
        f"see {server.log_path} for details."
    )


def add_server_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: str,
    default_base_url: str = DEFAULT_BASE_URL,
) -> None:
    """Register the flags :func:`ensure_server` reads."""
    parser.add_argument(
        "--base-url",
        default=default_base_url,
        help="vLLM server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help=(
            "Model id as registered with the server (matches /v1/models). "
            "Also used to launch the server when auto-starting it. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--reasoning-model",
        dest="is_reasoning_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether the server was (or should be) started with "
            "--cohere-is-reasoning-model (default true). Controls whether "
            "the thinking cases run, and whether an auto-started server "
            "gets --no-cohere-is-reasoning-model."
        ),
    )
    parser.add_argument(
        "--auto-start-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Launch 'vllm serve --model <model>' ourselves if --base-url "
            "isn't already serving a healthy /health response (default true)."
        ),
    )
    parser.add_argument(
        "--keep-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Leave an auto-started server running after the tests finish, "
            "so re-runs skip the model load (default true). Only applies "
            "when this script started the server itself."
        ),
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=1800.0,
        help=(
            "Seconds to wait for an auto-started server to become healthy "
            "(default: %(default)s). Large quantized checkpoints can take "
            "a long time to load."
        ),
    )
    parser.add_argument(
        "--extra-server-arg",
        dest="extra_server_args",
        action="append",
        default=[],
        help=(
            "Extra 'vllm serve' argument, forwarded verbatim when this "
            "script starts the server itself. Repeatable, e.g. "
            "--extra-server-arg=--tensor-parallel-size=8"
        ),
    )


def ensure_server(
    args: argparse.Namespace,
    *,
    log_prefix: str = "vllm-cohere-e2e-",
) -> ManagedServer | None:
    """Make ``args.base_url`` serviceable, starting a server if needed.

    Returns the :class:`ManagedServer` we own and the caller must release
    via :func:`release_server`, or ``None`` when the URL was already
    healthy (or auto-start was declined, which only warns -- the tests
    then fail on their own and say why).
    """
    if not args.auto_start_server:
        if not server_is_healthy(args.base_url):
            print(
                f"WARNING: {args.base_url} does not look healthy and "
                f"--no-auto-start-server was passed; tests will likely fail."
            )
        return None

    if server_is_healthy(args.base_url):
        return None

    parsed = urlparse(args.base_url)
    server = start_server(
        model=args.model,
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 8000,
        is_reasoning_model=args.is_reasoning_model,
        extra_args=args.extra_server_args,
        log_prefix=log_prefix,
    )
    try:
        wait_until_healthy(
            base_url=args.base_url,
            server=server,
            timeout=args.startup_timeout,
        )
    except Exception:
        server.stop()
        raise
    return server


def release_server(server: ManagedServer | None, *, keep: bool) -> None:
    """Stop a server we started, or explain how to stop it later."""
    if server is None:
        return
    if keep:
        print(
            f"\nLeaving auto-started server running "
            f"(pid={server.proc.pid}, log={server.log_path}).\n"
            f"Stop it with: kill {server.proc.pid}"
        )
    else:
        print(f"\nStopping auto-started server (pid={server.proc.pid})...")
        server.stop()
