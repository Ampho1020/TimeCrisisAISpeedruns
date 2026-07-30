"""Line-protocol TCP client for bizhawk_bridge.lua."""

import socket

from config import GUNCON_CALIB


def apply_guncon_calibration(aim_x, aim_y):
    """Map normalized [0,1] aim through the Guncon calibration transform.

    Corrects the edge drift seen with the Nymashock Guncon (no built-in
    offset/scale UI). Applied about screen center; see GUNCON_CALIB in
    config.py for the tuned values.
    """
    c = GUNCON_CALIB
    x = c["center_x"] + (aim_x - c["center_x"]) * c["scale_x"] + c["offset_x"]
    y = c["center_y"] + (aim_y - c["center_y"]) * c["scale_y"] + c["offset_y"]
    # clamp so we never send out-of-range coordinates to the port
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    return x, y


class BridgeClient:
    """
    Protocol (one command per line, one reply per line):
        read_u16 <addr>                        -> OK <value>
        set_input <shoot01> <cover01> <bias>   -> OK
        step <n>                               -> OK
        load <slot>                            -> OK
        save <slot>                            -> OK
        frame                                  -> OK <framecount>
        hud <line1|line2|...>                  -> OK
        hud_clear                              -> OK
    Errors come back as: ERR <message>
    """

    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.file = None

    # -- lifecycle ------------------------------------------------------

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.file = self.sock.makefile("rwb")

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
        self.file = None
        self.sock = None

    # -- transport ------------------------------------------------------

    def _cmd(self, line: str):
        if self.file is None:
            raise RuntimeError("Bridge not connected. Call connect() first.")
        self.file.write((line + "\n").encode("utf-8"))
        self.file.flush()
        raw = self.file.readline()
        if not raw:
            raise RuntimeError("Bridge disconnected (is the Lua script still running?)")
        text = raw.decode("utf-8", errors="replace").strip()
        if not text.startswith("OK"):
            raise RuntimeError(f"Bridge error for '{line}': {text}")
        parts = text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else None

    # -- commands -------------------------------------------------------

    def read_u16(self, addr: int) -> int:
        return int(self._cmd(f"read_u16 0x{addr:X}"))

    def set_input(self, shoot: bool, cover: bool, aim_bias: float = 0.0):
        self._cmd(f"set_input {1 if shoot else 0} {1 if cover else 0} {aim_bias:.4f}")

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
