"""Utilities for launching Ark debug tooling alongside Newton scripts.

This module provides light wrappers for spawning ``lcm-logger`` and
``ark_graph`` in separate processes so that developers can inspect the
communication graph and capture LCM traffic while running local scripts.
"""

from __future__ import annotations

import atexit
import subprocess
import time
from multiprocessing import Process
from pathlib import Path
from typing import Optional

from ark.tools.log import log


class LcmLoggerSession:
    """Manage a background ``lcm-logger`` process."""

    def __init__(self, log_path: Path | str, auto_start: bool = True) -> None:
        self.log_path = Path(log_path).expanduser().resolve()
        self.proc: Optional[subprocess.Popen[bytes]] = None

        if auto_start:
            self.start()

        atexit.register(self.stop)

    def start(self) -> None:
        """Start ``lcm-logger`` unless it is already running."""
        if self.proc is not None:
            log.warning("lcm-logger already running for %s", self.log_path)
            return

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["lcm-logger", str(self.log_path)]

        try:
            self.proc = subprocess.Popen(  # noqa: S603, S607 (external tool)
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            log.error("Could not find lcm-logger executable on PATH.")
            self.proc = None
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to launch lcm-logger: %s", exc)
            self.proc = None
        else:
            log.ok("lcm-logger capturing to %s", self.log_path)

    def stop(self) -> None:
        """Terminate the logger process if active."""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception as exc:  # noqa: BLE001
                log.warning("Error while stopping lcm-logger: %s", exc)
            else:
                log.info("lcm-logger stopped")
        self.proc = None


def _launch_ark_graph(
    host: str,
    port: int,
    delay: float,
    display: bool,
    snapshot_path: Optional[str],
) -> None:
    """Worker process that instantiates :class:`ArkGraph`."""

    try:
        if delay > 0.0:
            time.sleep(delay)

        from ark.tools.ark_graph.ark_graph import ArkGraph

        graph = ArkGraph(registry_host=host, registry_port=port, display=display)
        if snapshot_path:
            try:
                graph.save_image(snapshot_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[ArkGraphMonitor] Failed to save snapshot: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ArkGraphMonitor] Failed to launch ArkGraph: {exc}")


class ArkGraphMonitor:
    """Spawn ``ArkGraph`` in a separate process so the main script can continue."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1234,
        delay: float = 3.0,
        display: bool = True,
        snapshot_path: Path | str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.delay = delay
        self.display = display
        self.snapshot_path = str(snapshot_path) if snapshot_path else None
        self._process: Optional[Process] = None

    def start(self) -> None:
        """Start the ArkGraph process."""
        if self._process is not None and self._process.is_alive():
            log.warning("ArkGraph monitor already running")
            return

        self._process = Process(
            target=_launch_ark_graph,
            args=(self.host, self.port, self.delay, self.display, self.snapshot_path),
            daemon=True,
        )
        self._process.start()
        log.info(
            "Started ArkGraph monitor (host=%s, port=%d, delay=%.1fs)",
            self.host,
            self.port,
            self.delay,
        )
        atexit.register(self.stop)

    def stop(self) -> None:
        """Terminate the monitor process if it is still running."""
        if self._process is None:
            return
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=1)
        self._process = None


class DebugSession:
    """Context manager bundling both logger and graph monitors."""

    def __init__(
        self,
        enable_logger: bool,
        log_path: Path | str,
        enable_graph: bool,
        host: str,
        port: int,
        graph_delay: float,
        snapshot_path: Path | str | None = None,
    ) -> None:
        self.enable_logger = enable_logger
        self.enable_graph = enable_graph
        self.log_path = log_path
        self.host = host
        self.port = port
        self.graph_delay = graph_delay
        self.snapshot_path = snapshot_path

        self._logger: Optional[LcmLoggerSession] = None
        self._graph: Optional[ArkGraphMonitor] = None

    def __enter__(self) -> "DebugSession":
        if self.enable_logger:
            self._logger = LcmLoggerSession(self.log_path)
        if self.enable_graph:
            self._graph = ArkGraphMonitor(
                host=self.host,
                port=self.port,
                delay=self.graph_delay,
                snapshot_path=self.snapshot_path,
            )
            self._graph.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401, ANN001
        if self._graph is not None:
            self._graph.stop()
        if self._logger is not None:
            self._logger.stop()


__all__ = [
    "ArkGraphMonitor",
    "DebugSession",
    "LcmLoggerSession",
]
