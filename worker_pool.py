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
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from config import (
    AUTO_LAUNCH_BIZHAWK, BASE_PORT, BIZHAWK_EXTRA_ARGS, BIZHAWK_LAUNCH,
    BIZHAWK_LAUNCH_STAGGER_SECONDS, BIZHAWK_LUA, BIZHAWK_ROM, HOST,
    NUM_WORKERS,
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
    # Keep the trainer log readable: BizHawk itself is extremely chatty
    # (state-load timing lines every reset), which can drown ES progress output.
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
        self._closing = False

    # -- lifecycle ------------------------------------------------------

    def start(self):
        """Bind all listeners, (optionally) launch all emulators, then connect."""
        # 1. Bind every listener FIRST so no emulator races ahead of its port.
        for env in self.envs:
            env.start_listening()

        # 2. Bring up the emulators. Staggered so concurrent instances don't
        #    race on BizHawk's single shared config.ini during startup (see
        #    BIZHAWK_LAUNCH_STAGGER_SECONDS in config.py).
        if AUTO_LAUNCH_BIZHAWK:
            self._procs = []
            for idx, p in enumerate(self.ports):
                self._procs.append(_launch_bizhawk(p))
                if idx < len(self.ports) - 1 and BIZHAWK_LAUNCH_STAGGER_SECONDS > 0:
                    time.sleep(BIZHAWK_LAUNCH_STAGGER_SECONDS)
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
        # Signal in-flight evals to stop recovering: without this, threads that
        # are mid-episode when we tear down catch the resulting socket errors
        # and try to "recover" by relaunching BizHawk, leaving orphan emulators
        # behind (and racing with the list teardown below).
        self._closing = True
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

    def _restart_worker(self, worker_idx: int):
        """Tear down and rebuild ONE flaky worker's emulator + bridge in place.

        A single BizHawk instance occasionally drops its socket mid-run (e.g.
        the bridge returns no value for a ``read_u16_multi`` after the emulator
        wedges on a continue/menu screen). Without recovery that one dropout
        raises straight through ``evaluate()`` and aborts the entire training
        run, throwing away every generation so far. This recycles only the
        affected port -- the other workers keep their live connections -- so
        the run can carry on. Each worker index is driven by exactly one
        thread, so mutating ``self._procs[worker_idx]``/``self.envs[worker_idx]``
        here is race-free.
        """
        # If the pool is tearing down, do NOT rebuild: close() may have already
        # cleared self._procs (the assignment below would IndexError) and any
        # emulator we launch now would be orphaned past shutdown.
        if self._closing:
            raise RuntimeError("worker pool is closing; skipping worker restart")
        port = self.ports[worker_idx]
        env = self.envs[worker_idx]
        print(
            f"[pool] restarting worker {worker_idx} on port {port} "
            f"after a bridge failure...",
            flush=True,
        )
        # Drop the old bridge sockets (best-effort).
        try:
            env.close()
        except Exception:
            pass
        # Kill the old emulator process, if we launched it.
        if AUTO_LAUNCH_BIZHAWK and worker_idx < len(self._procs):
            old = self._procs[worker_idx]
            try:
                old.terminate()
                old.wait(timeout=10)
            except Exception:
                try:
                    old.kill()
                except Exception:
                    pass
        # Rebind the listener BEFORE relaunching so the emulator can dial in.
        env.start_listening()
        if AUTO_LAUNCH_BIZHAWK:
            proc = _launch_bizhawk(port)
            # Index-safe: a concurrent close() can shrink/clear self._procs
            # while this restart is in flight -- grow it rather than IndexError.
            while len(self._procs) <= worker_idx:
                self._procs.append(None)
            self._procs[worker_idx] = proc
        else:
            print(
                f"[pool] AUTO_LAUNCH_BIZHAWK is off -- relaunch BizHawk on "
                f"port {port} now so worker {worker_idx} can reconnect.",
                flush=True,
            )
        # Block until it reconnects + completes the Lua handshake.
        env.finish_connect()
        print(f"[pool] worker {worker_idx} back online.", flush=True)

    def _eval_with_recovery(self, worker_idx: int, candidate, max_restarts: int = 2):
        """Evaluate one candidate, restarting the worker and retrying on a
        bridge/emulator failure. Only raises (aborting the run) if the worker
        cannot be brought back after ``max_restarts`` attempts -- i.e. a truly
        dead setup, not a transient blip."""
        attempt = 0
        while True:
            try:
                return self.envs[worker_idx].episode_fitness(candidate)
            except Exception as exc:
                # Don't try to recover a failure that is just the pool being
                # torn down -- that only spawns orphan emulators mid-shutdown.
                if self._closing:
                    raise
                attempt += 1
                print(
                    f"[pool] worker {worker_idx} eval failed "
                    f"(attempt {attempt}/{max_restarts}): {exc!r}",
                    flush=True,
                )
                if attempt > max_restarts:
                    print(
                        f"[pool] worker {worker_idx} unrecoverable after "
                        f"{max_restarts} restart(s) -- aborting.",
                        flush=True,
                    )
                    raise
                try:
                    self._restart_worker(worker_idx)
                except Exception as rexc:
                    print(
                        f"[pool] worker {worker_idx} restart failed: {rexc!r}",
                        flush=True,
                    )
                    raise

    def evaluate(self, candidates, progress_cb=None):
        """Evaluate every candidate; returns aligned [(fitness, info), ...].

        Candidates are dealt round-robin to workers, so each env is touched by
        exactly one thread at a time (no shared-socket races), and the load is
        balanced even when len(candidates) isn't a multiple of num_workers.
        """
        if self._executor is None:
            raise RuntimeError("WorkerPool.start() must be called before evaluate().")

        results: list = [None] * len(candidates)
        done = 0
        done_lock = threading.Lock()

        def run_chunk(worker_idx: int):
            nonlocal done
            for i in range(worker_idx, len(candidates), self.num_workers):
                results[i] = self._eval_with_recovery(worker_idx, candidates[i])
                if progress_cb is not None:
                    with done_lock:
                        done += 1
                        progress_cb(done, len(candidates))

        # One task per worker; each drains its slice sequentially on its own env.
        list(self._executor.map(run_chunk, range(self.num_workers)))
        return results
