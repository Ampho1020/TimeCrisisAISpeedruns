"""Parallel evaluation across several BizHawk instances.

Each worker owns one BizHawk emulator (on its own socket port) and one
TimeCrisisEnv. A generation's candidates are split across the workers and
evaluated concurrently. Evaluation blocks on socket I/O waiting for the
emulator, so plain threads (which release the GIL during recv) give real
parallelism here -- no multiprocessing needed.

Launch ordering matters: Python must be listening on a port BEFORE the BizHawk
that dials into it starts. So we bind every listener first, then launch (or wait
for) every emulator, then accept + handshake each.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor

from config import (
    AUTO_LAUNCH_BIZHAWK, BASE_PORT, BIZHAWK_EXTRA_ARGS, BIZHAWK_LAUNCH,
    BIZHAWK_LUA, BIZHAWK_ROM, HOST, NUM_WORKERS,
)
from env_timecrisis import TimeCrisisEnv


def _launch_bizhawk(port: int) -> subprocess.Popen:
    """Spawn one EmuHawk wired to `port`, auto-loading the Lua bridge."""
    if not BIZHAWK_ROM:
        raise RuntimeError(
            "AUTO_LAUNCH_BIZHAWK is on but BIZHAWK_ROM is empty. Set the disc "
            "image path (and BIZHAWK_LAUNCH) in config.py, or launch the "
            "emulators yourself and set AUTO_LAUNCH_BIZHAWK = False."
        )
    cmd = [
        BIZHAWK_LAUNCH,
        BIZHAWK_ROM,
        f"--socket_ip={HOST}",
        f"--socket_port={port}",
        f"--lua={BIZHAWK_LUA}",
        *BIZHAWK_EXTRA_ARGS,
    ]
    print(f"[pool] launching BizHawk on port {port}: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd)


class WorkerPool:
    """Owns N emulator-backed envs and evaluates populations across them."""

    def __init__(self, num_workers: int = NUM_WORKERS, base_port: int = BASE_PORT):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.num_workers = num_workers
        self.ports = [base_port + i for i in range(num_workers)]
        self.envs = [TimeCrisisEnv(host=HOST, port=p) for p in self.ports]
        self._procs: list[subprocess.Popen] = []
        self._executor: ThreadPoolExecutor | None = None

    # -- lifecycle ------------------------------------------------------

    def start(self):
        """Bind all listeners, (optionally) launch all emulators, then connect."""
        # 1. Bind every listener FIRST so no emulator races ahead of its port.
        for env in self.envs:
            env.start_listening()

        # 2. Bring up the emulators.
        if AUTO_LAUNCH_BIZHAWK:
            self._procs = [_launch_bizhawk(p) for p in self.ports]
        else:
            print(
                f"[pool] AUTO_LAUNCH_BIZHAWK is off. Launch {self.num_workers} "
                f"BizHawk instance(s) now, one per port: {self.ports}\n"
                f"       Each with --socket_ip={HOST} --socket_port=<port> "
                f"--lua={BIZHAWK_LUA}",
                flush=True,
            )

        # 3. Accept + handshake each (blocks per worker until its emulator dials
        #    in; order-independent since each port has its own listener).
        for i, env in enumerate(self.envs):
            print(f"[pool] waiting for worker {i} on port {self.ports[i]}...", flush=True)
            env.finish_connect()

        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        print(f"[pool] {self.num_workers} worker(s) live.", flush=True)

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
        for proc in self._procs:
            try:
                proc.terminate()
            except Exception:
                pass
        self._procs = []

    # -- evaluation -----------------------------------------------------

    def evaluate(self, candidates):
        """Evaluate every candidate; returns aligned [(fitness, info), ...].

        Candidates are dealt round-robin to workers, so each env is touched by
        exactly one thread at a time (no shared-socket races), and the load is
        balanced even when len(candidates) isn't a multiple of num_workers.
        """
        if self._executor is None:
            raise RuntimeError("WorkerPool.start() must be called before evaluate().")

        results: list = [None] * len(candidates)

        def run_chunk(worker_idx: int):
            env = self.envs[worker_idx]
            for i in range(worker_idx, len(candidates), self.num_workers):
                results[i] = env.episode_fitness(candidates[i])

        # One task per worker; each drains its slice sequentially on its own env.
        list(self._executor.map(run_chunk, range(self.num_workers)))
        return results
