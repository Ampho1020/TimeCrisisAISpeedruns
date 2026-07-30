"""Line-protocol server for bizhawk_bridge.lua using BizHawk 2.11.1 comm.*.

BizHawk's comm.socketServer* transport is the reverse of the old LuaSocket
bridge: the Python side listens, then the Lua script running in BizHawk
connects out to it. Once connected, the logical protocol remains one command
per line and one reply per line.
"""

from __future__ import annotations

import socket

from config import GUNCON_CALIB


def apply_guncon_calibration(aim_x, aim_y):
    """Map normalized [0,1] aim through the Guncon calibration transform."""
    c = GUNCON_CALIB
    x = c["center_x"] + (aim_x - c["center_x"]) * c["scale_x"] + c["offset_x"]
    y = c["center_y"] + (aim_y - c["center_y"]) * c["scale_y"] + c["offset_y"]
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    return x, y


class BridgeClient:
    """
    Protocol (one command per line, one reply per line):
        read_u16 <addr>                            -> OK <value>
        set_input <shoot01> <cover01> <aim_x> <aim_y> -> OK
        step <n>                                   -> OK
        load <slot>                                -> OK
        save <slot>                                -> OK
        frame                                      -> OK <framecount>
        hud <line1|line2|...>                      -> OK
        hud_clear                                  -> OK

    Backward compatibility note:
        Legacy callers may still pass aim_bias without aim_x/aim_y. In that
        case the bias is mapped onto a centered X-only placeholder aim until
        real vision-based X/Y aiming lands.
    """

    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.server_sock = None
        self.sock = None
        self.file = None

    # -- lifecycle ------------------------------------------------------

    def connect(self):
        self.close()
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(1)
        self.server_sock.settimeout(self.timeout)
        self.sock, _ = self.server_sock.accept()
        self.sock.settimeout(self.timeout)
        self.file = self.sock.makefile("rwb")
        self.server_sock.close()
        self.server_sock = None

    def close(self):
        try:
            if self.file:
                self.file.close()
        except Exception:
            pass
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        try:
            if self.server_sock:
                self.server_sock.close()
        except Exception:
            pass
        self.file = None
        self.sock = None
        self.server_sock = None

    # -- transport ------------------------------------------------------

    def _cmd(self, line: str):
        if self.file is None:
            raise RuntimeError(
                "Bridge not connected. Start the Python process so it is listening, "
                "then load bizhawk_bridge.lua in BizHawk to connect out."
            )
        self.file.write((line + "\n").encode("utf-8"))
        self.file.flush()
        raw = self.file.readline()
        if not raw:
            raise RuntimeError("Bridge disconnected (is bizhawk_bridge.lua still running?)")
        text = raw.decode("utf-8", errors="replace").strip()
        if not text.startswith("OK"):
            raise RuntimeError(f"Bridge error for '{line}': {text}")
        parts = text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else None

    # -- commands -------------------------------------------------------

    def read_u16(self, addr: int) -> int:
        return int(self._cmd(f"read_u16 0x{addr:X}"))

    def set_input(
        self,
        shoot: bool,
        cover: bool,
        aim_bias: float | None = None,
        *,
        aim_x: float | None = None,
        aim_y: float | None = None,
    ):
        if aim_x is None or aim_y is None:
            # Backward-compatible fallback while policy.py still emits a single
            # aim_bias scalar. Real X/Y aiming should pass aim_x/aim_y instead.
            legacy_bias = 0.0 if aim_bias is None else float(aim_bias)
            legacy_bias = min(1.0, max(-1.0, legacy_bias))
            aim_x = 0.5 + 0.5 * legacy_bias
            aim_y = 0.5

        aim_x, aim_y = apply_guncon_calibration(float(aim_x), float(aim_y))
        self._cmd(f"set_input {1 if shoot else 0} {1 if cover else 0} {aim_x:.4f} {aim_y:.4f}")

    def step_frames(self, n: int = 1):
        self._cmd(f"step {int(n)}")

    def load_state(self, slot: int = 1):
        self._cmd(f"load {int(slot)}")

    def save_state(self, slot: int = 1):
        self._cmd(f"save {int(slot)}")

    def frame(self) -> int:
        return int(self._cmd("frame"))

    def hud(self, lines):
        safe = [str(s).replace("|", "/").replace("\n", " ") for s in lines]
        self._cmd("hud " + "|".join(safe))

    def hud_clear(self):
        self._cmd("hud_clear")
