"""Line-protocol server for bizhawk_bridge.lua (BizHawk 2.11.1 comm.* model).

TRANSPORT DIRECTION (important):
    Under BizHawk's comm.socketServer* API, BizHawk is the socket *client*: it
    dials OUT to an external listener the moment it launches (it connects in its
    MainForm constructor). Therefore THIS class must be the *server* -- it binds
    the port, listens, and accepts the connection BizHawk makes.

STARTUP ORDER (must be followed or BizHawk crashes with "Connection refused"):
    1. Start Python first (this creates the listener and blocks on accept()):
           python es_train.py
    2. Only then launch BizHawk WITH the socket flags:
           ./EmuHawk --socket_ip=127.0.0.1 --socket_port=8765
    3. Load the game (Guncon port, savestate slot 1), then open bizhawk_bridge.lua.

    If Python is not already listening when BizHawk launches, BizHawk's launch-
    time connect is refused and it crashes in MainForm..ctor.

HANDSHAKE:
    BizHawk's socket connects at launch, but only the Lua script answers
    commands. So accept() can return long before the script is loaded. To avoid
    the first command timing out, the Lua script sends a single "READY" line on
    load; connect() below blocks (with a long timeout) until it reads that line
    before returning. This also consumes the READY so it can't corrupt the first
    real command/reply pair.

Protocol (one command per line, one reply per line):
    read_u16 <addr>                                 -> OK <value>
    set_input <shoot01> <cover01> <aim_x> <aim_y>   -> OK
    step <n>                                         -> OK
    load <slot>                                      -> OK
    save <slot>                                      -> OK
    frame                                            -> OK <framecount>
    hud <line1|line2|...>                            -> OK
    hud_clear                                        -> OK
Errors come back as: ERR <message>
"""

import socket

from config import GUNCON_CALIB

# How long to wait for the Lua "READY" handshake after BizHawk connects. Long,
# because the user still has to load the game and open the Lua script.
HANDSHAKE_TIMEOUT = 120.0


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
    """Server side of the bridge. Named 'BridgeClient' for backwards
    compatibility with env_timecrisis.py / es_train.py -- the public method
    surface (connect/close/read_u16/set_input/step_frames/load_state/
    save_state/frame/hud/hud_clear) is unchanged; only the transport flipped
    from dial-out client to listen-and-accept server.
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
        """Bind, listen, accept, then block until the Lua READY handshake.

        Call this BEFORE launching BizHawk. It blocks on accept() until the
        emulator dials in, then blocks reading until the Lua script sends
        "READY" (so training never fires a command before the script is live).
        """
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(1)
        print(
            f"[bridge] listening on {self.host}:{self.port} -- "
            f"now launch BizHawk with --socket_ip={self.host} --socket_port={self.port}",
            flush=True,
        )
        # Block until BizHawk connects (no timeout on the initial accept, since
        # the user may take a moment to launch the emulator).
        self.sock, addr = self.server_sock.accept()
        self.file = self.sock.makefile("rwb")
        print(
            f"[bridge] BizHawk connected from {addr}; "
            f"waiting for Lua READY (load the game and open bizhawk_bridge.lua)...",
            flush=True,
        )

        # Wait for the Lua script to announce it is live. Use a long timeout for
        # the handshake, then drop back to the normal per-command timeout.
        self.sock.settimeout(HANDSHAKE_TIMEOUT)
        while True:
            raw = self.file.readline()
            if not raw:
                raise RuntimeError(
                    "Connection closed before READY handshake "
                    "(did the Lua script fail to load?)."
                )
            if raw.decode("utf-8", errors="replace").strip() == "READY":
                break
            # Ignore any other chatter until READY arrives.

        self.sock.settimeout(self.timeout)
        print("[bridge] handshake OK -- bridge live", flush=True)

    def close(self):
        for obj in (self.file, self.sock, self.server_sock):
            try:
                if obj:
                    obj.close()
            except Exception:
                pass
        self.file = None
        self.sock = None
        self.server_sock = None

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

    def set_input(self, shoot: bool, cover: bool, aim_x: float = 0.5, aim_y: float = 0.5):
        # Apply the Guncon calibration exactly once, right before sending.
        cx, cy = apply_guncon_calibration(aim_x, aim_y)
        self._cmd(
            f"set_input {1 if shoot else 0} {1 if cover else 0} {cx:.4f} {cy:.4f}"
        )

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
