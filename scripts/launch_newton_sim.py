#!/usr/bin/env python3
"""Launch ARK simulator with Newton physics backend for Franka Panda.

This script starts the ARK simulator node using the Newton backend and can
optionally spawn auxiliary tooling that helps debug the message graph and
captures LCM traffic for later inspection.

Usage (default configuration):
    python scripts/launch_newton_sim.py

With debug helpers enabled/disabled explicitly:
    python scripts/launch_newton_sim.py --log-file logs/run.lcm --graph-delay 8
    python scripts/launch_newton_sim.py --no-graph --no-logger
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# Ensure we can import from the local utils folder
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ark.system.simulation.simulator_node import SimulatorNode
from ark.tools.log import log

from utils.debug_tools import DebugSession

# Path to Newton configuration (relative to ark_franka/)
ROOT = Path(__file__).resolve().parents[1]  # ark_franka/
CONFIG_PATH = str(ROOT / "tests" / "config" / "global_config_newton.yaml")
DEFAULT_LOG_PATH = ROOT / "logs" / "newton_session.lcm"


class NewtonSimulatorNode(SimulatorNode):
    """ARK Simulator Node using Newton physics backend."""

    def __init__(self, *args, **kwargs):
        """Initialize with flag to track if first step_physics call."""
        self._init_physics_done = False
        super().__init__(*args, **kwargs)

    def step_physics(self, num_steps: int = 25, call_step_hook: bool = False) -> None:
        """Override to skip initial 25-step burst during initialization.

        The base SimulatorNode.__init__ calls step_physics(25) immediately after
        initialize_scene(). For Newton/XPBD, this can cause the robot to explode
        before proper warmup if joint configuration isn't fully applied yet.
        """
        if not self._init_physics_done:
            # Skip the initial burst during __init__, but mark as done for future calls
            self._init_physics_done = True
            log.info("Newton: Skipping initial step_physics burst during initialization")
            return
        super().step_physics(num_steps, call_step_hook)

    def initialize_scene(self):
        """Called after simulator backend is initialized.

        This is where you can add custom initialization logic,
        spawn additional objects, or configure the scene.
        """
        log.ok("=" * 60)
        log.ok("Newton Simulator Initialized")
        log.ok("=" * 60)
        log.info(f"Backend type: {self.backend.__class__.__name__}")
        log.info(f"Robots loaded: {list(self.backend.robot_ref.keys())}")
        log.info(f"Objects loaded: {list(self.backend.object_ref.keys())}")
        log.info(f"Sensors loaded: {list(self.backend.sensor_ref.keys())}")
        log.ok("=" * 60)
        log.ok("Simulator ready. Send commands via ARK channels:")
        log.info("  Joint commands: panda/joint_group_command/sim")
        log.info("  Joint states: panda/joint_states/sim")
        log.ok("=" * 60)

    def step(self):
        """Called each simulation step.

        Override this to add custom per-step logic like:
        - Logging state
        - Custom forces
        - Triggering events
        """
        pass


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for enabling debug helpers."""
    parser = argparse.ArgumentParser(description="Launch Newton-backed ARK simulator")
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH,
        help="Path to simulator global configuration YAML",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(DEFAULT_LOG_PATH),
        help="Output path for lcm-logger capture (.lcm file)",
    )
    parser.add_argument(
        "--logger",
        dest="enable_logger",
        action="store_true",
        help="Enable lcm-logger capture (default)",
    )
    parser.add_argument(
        "--no-logger",
        dest="enable_logger",
        action="store_false",
        help="Disable lcm-logger capture",
    )
    parser.add_argument(
        "--graph",
        dest="enable_graph",
        action="store_true",
        help="Launch ArkGraph viewer (default)",
    )
    parser.add_argument(
        "--no-graph",
        dest="enable_graph",
        action="store_false",
        help="Skip ArkGraph viewer",
    )
    parser.add_argument(
        "--graph-host",
        type=str,
        default="127.0.0.1",
        help="Registry host ArkGraph should query",
    )
    parser.add_argument(
        "--graph-port",
        type=int,
        default=1234,
        help="Registry port ArkGraph should query",
    )
    parser.add_argument(
        "--graph-delay",
        type=float,
        default=5.0,
        help="Seconds to wait before ArkGraph connects (allows simulator registration)",
    )
    parser.add_argument(
        "--graph-snapshot",
        type=str,
        default=None,
        help="Optional path to save ArkGraph PNG snapshot",
    )

    parser.set_defaults(enable_logger=True, enable_graph=True)
    return parser.parse_args()


def run() -> None:
    """Entry point used by ``__main__`` guard."""
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    log.info(f"Starting Newton simulator with config: {config_path}")

    if args.enable_logger:
        log.info(f"LCM traffic will be recorded to {args.log_file}")
    if args.enable_graph:
        log.info(
            "ArkGraph viewer launching after %.1fs (registry %s:%d)",
            args.graph_delay,
            args.graph_host,
            args.graph_port,
        )

    with DebugSession(
        enable_logger=args.enable_logger,
        log_path=args.log_file,
        enable_graph=args.enable_graph,
        host=args.graph_host,
        port=args.graph_port,
        graph_delay=args.graph_delay,
        snapshot_path=args.graph_snapshot,
    ):
        node: NewtonSimulatorNode | None = None
        try:
            log.ok(f"Initializing {NewtonSimulatorNode.__name__} node")
            node = NewtonSimulatorNode(str(config_path))
            log.ok(f"Initialized {node.name}")

            # Run the simulation loop with proper LCM handling.
            # SimulatorNode.spin() processes LCM messages and will invoke the
            # backend custom loop (e.g., Newton viewer) when available.
            node.spin()
        except KeyboardInterrupt:
            log.warning("User interrupted Newton simulator")
        except Exception:
            tb = traceback.format_exc()
            div = "=" * 30
            log.error(f"Exception thrown during node execution:\n{div}\n{tb}\n{div}")
        finally:
            if node is not None:
                node.kill_node()
                log.ok(f"Finished running node {NewtonSimulatorNode.__name__}")
            else:
                log.warning(f"{NewtonSimulatorNode.__name__} failed during initialization")
        # Ensure we exit cleanly after kill_node (which may already terminate the process)
        sys.exit(0)


if __name__ == "__main__":
    run()
