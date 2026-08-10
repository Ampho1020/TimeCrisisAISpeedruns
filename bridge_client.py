"""Length-prefixed socket bridge for bizhawk_bridge.lua (BizHawk 2.11.1 comm.*).

TRANSPORT DIRECTION (important):
    Under BizHawk's comm.socketServer* API, BizHawk is the socket *client*: it
    dials OUT to an external listener the moment it launches. Therefore THIS
    class is the *server* -- it binds the port, listens, and accepts BizHawk.

WIRE FORMAT (critical -- verified against BizHawk source):
    Since BizHawk 2.6.2, comm.socketServerResponse() does NOT read newline-
    terminated lines. Every message in BOTH directions is length-prefixed:

        "{N} {payload}"

    where N is the byte length of payload in base-10 followed by a single space.
    e.g. to send "ping" you must put "4 ping" on the wire; BizHawk replies the
    same way, e.g. "10 PONG:ping". We therefore frame every outbound command and
    parse every inbound reply using this decimal-length prefix -- NOT newlines.

    (This was the root cause of the long "messages never arrive" saga: we were
    sending newline-terminated text, which BizHawk's receiver silently drops.)

STARTUP ORDER (must be followed or BizHawk crashes with "Connection refused"):
    1. Start Python first (binds the port, waits on accept):
           python es_train.py
    2. Only then launch BizHawk WITH the socket flags:
           ./EmuHawk --socket_ip=127.0.0.1 --socket_port=8765
    3. Load the game (Guncon port, savestate slot 1), then open bizhawk_bridge.lua.

HANDSHAKE (Python-initiated):
    BizHawk's socket connects at launch, but only the Lua script answers
    commands, and on this build socketServerResponse() only *sends* as the reply
    half of a *received* message -- Lua cannot reliably speak first. So the Lua
    script can't just emit READY unprompted. Instead, connect() POLLS: it sends
    "hello" repeatedly (short per-send timeout) until the Lua script -- once
    loaded -- replies "READY". This mirrors the proven ping/pong test and cleanly
    handles the fact that the script is opened after BizHawk launches.

Commands (payload text, before framing):
    read_u16 <addr>                                 -> OK <value>
    set_input <shoot01> <cover01> <aim_x> <aim_y>   -> OK
    step <n>                                         -> OK
    load <slot>                                      -> OK
    save <slot>                                      -> OK
    frame                                            -> OK <framecount>
    hud <line1|line2|...>                            -> OK
    hud_clear                                        -> OK
    screenshot                                       -> (raw BMP bytes, length-
                                                        prefixed just like every
                                                        other reply; NO trailing
                                                        OK -- see get_screenshot)
Errors come back as: ERR <message>
"""

import socket

import numpy as np

from config import GUNCON_CALIB
from vision import decode_bmp

# How long to poll for the Lua handshake after BizHawk connects. Long, because
# the user still has to load the game and open the Lua script.
HANDSHAKE_TIMEOUT = 120.0
# Per-attempt send/recv timeout while polling the handshake.
HANDSHAKE_POLL_INTERVAL = 0.5


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
    save_state/frame/hud/hud_clear) is unchanged.

    Uses raw recv/sendall (NOT socket.makefile) because a makefile object gets
    permanently poisoned if a read times out ("cannot read from timed out
    object"), which is fatal once we set per-read timeouts.
    """

    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.server_sock: socket.socket | None = None
        self.sock: socket.socket | None = None
        self._recv_buf = b""  # leftover bytes between framed reads

    # -- framing helpers ------------------------------------------------

    @staticmethod
    def _frame(payload: str) -> bytes:
        """Wrap a payload in BizHawk's "{len} {payload}" length prefix."""
        data = payload.encode("utf-8")
        return f"{len(data)} ".encode("utf-8") + data

    def _send(self, payload: str):
        if self.sock is None:
            raise RuntimeError("Bridge not connected. Call connect() first.")
        self.sock.sendall(self._frame(payload))

    def _recv_message_bytes(self) -> bytes:
        """Read one length-prefixed message and return the RAW payload bytes.

        Same wire format as ``_recv_message`` but skips the UTF-8 decode step
        so binary payloads (e.g. the BMP screenshot from BizHawk's
        ``comm.socketServerScreenShot``) round-trip byte-exact. All existing
        text commands still go through ``_recv_message`` and decode as UTF-8.
        """
        if self.sock is None:
            raise RuntimeError("Bridge not connected. Call connect() first.")
        sock = self.sock
        # 1. Read up to and including the space that terminates the length.
        while b" " not in self._recv_buf:
            chunk = sock.recv(4096)
            if chunk == b"":
                raise RuntimeError(
                    "Bridge disconnected (is the Lua script still running?)"
                )
            self._recv_buf += chunk

        length_str, _, rest = self._recv_buf.partition(b" ")
        try:
            n = int(length_str)
        except ValueError:
            raise RuntimeError(f"Malformed length prefix: {length_str!r}")

        # 2. Ensure we have the full payload of n bytes. Bigger recv chunk than
        # the text path because a screenshot BMP is ~256 KB, not 100 B.
        self._recv_buf = rest
        while len(self._recv_buf) < n:
            chunk = sock.recv(65536)
            if chunk == b"":
                raise RuntimeError(
                    "Bridge disconnected mid-message "
                    "(is the Lua script still running?)"
                )
            self._recv_buf += chunk

        payload = self._recv_buf[:n]
        self._recv_buf = self._recv_buf[n:]
        return payload

    def _recv_message(self) -> str:
        """Read one length-prefixed message and decode it as UTF-8 text."""
        return self._recv_message_bytes().decode("utf-8", errors="replace")

    # -- lifecycle ------------------------------------------------------

    def connect(self):
        """Bind + listen, then accept and handshake in one call.

        Convenience wrapper for the single-instance flow. For a PARALLEL launch,
        call start_listening() on every worker first, launch all the emulators,
        then finish_connect() on each -- Python must be listening before BizHawk
        dials in, so binding has to happen before the emulators start.
        """
        self.start_listening()
        self.finish_connect()

    def start_listening(self):
        """Bind and listen so BizHawk can connect out to us. Does not block."""
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(1)
        print(
            f"[bridge] listening on {self.host}:{self.port} -- "
            f"now launch BizHawk with --socket_ip={self.host} --socket_port={self.port}",
            flush=True,
        )

    def finish_connect(self):
        """Accept the BizHawk connection, then poll a Python-initiated handshake.

        Blocks on accept() until the emulator dials in, then repeatedly sends
        "hello" until the Lua script (opened after BizHawk launches) replies
        "READY", and finally drains the stale handshake replies.
        """
        if self.server_sock is None:
            raise RuntimeError(
                "start_listening() must be called before finish_connect()."
            )
        # Block until BizHawk connects (no timeout on the initial accept).
        self.sock, addr = self.server_sock.accept()
        self._recv_buf = b""
        print(
            f"[bridge] BizHawk connected from {addr}; "
            f"polling for Lua handshake (load the game and open bizhawk_bridge.lua)...",
            flush=True,
        )

        # Python-initiated handshake: spam "hello" until Lua answers "READY".
        # Short per-attempt timeout so a missed reply just triggers another send.
        self.sock.settimeout(HANDSHAKE_POLL_INTERVAL)
        import time

        deadline = time.monotonic() + HANDSHAKE_TIMEOUT
        attempts = 0
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Timed out waiting for Lua handshake. Is bizhawk_bridge.lua "
                    "open and the game running?"
                )
            attempts += 1
            try:
                self._send("hello")
            except socket.timeout:
                continue
            except Exception as e:
                raise RuntimeError(f"Handshake send failed: {e!r}")

            try:
                reply = self._recv_message()
            except socket.timeout:
                if attempts % 10 == 0:
                    print(
                        f"[bridge] sent {attempts} handshakes, still waiting for READY...",
                        flush=True,
                    )
                continue

            if reply.strip() == "READY":
                break
            # Ignore any other chatter until READY arrives.

        self.sock.settimeout(self.timeout)

        # RESYNC: the handshake spammed many "hello" messages while waiting for
        # the Lua script to load. Once loaded, the Lua loop answers each queued
        # "hello" with "ERR unknown_cmd" one-per-frame, so those stale replies
        # are now sitting in the stream ahead of any real command's reply. If we
        # don't drain them, the first real command (e.g. "load 1") reads a stale
        # "ERR unknown_cmd" and wrongly fails. Send a sentinel command and
        # discard everything up to its "OK" reply. TCP is FIFO and the Lua loop
        # is single-threaded, so the sentinel's OK is guaranteed to arrive after
        # every stale hello->ERR; only hellos were ever sent, so the first "OK"
        # is unambiguously the sentinel's.
        self._send("frame")
        while True:
            resp = self._recv_message().strip()
            if resp.startswith("OK"):
                break

        print("[bridge] handshake OK -- bridge live", flush=True)

    def close(self):
        for obj in (self.sock, self.server_sock):
            try:
                if obj:
                    obj.close()
            except Exception:
                pass
        self.sock = None
        self.server_sock = None
        self._recv_buf = b""

    # -- transport ------------------------------------------------------

    def _cmd(self, line: str):
        if self.sock is None:
            raise RuntimeError("Bridge not connected. Call connect() first.")
        self._send(line)
        text = self._recv_message().strip()
        if not text.startswith("OK"):
            raise RuntimeError(f"Bridge error for '{line}': {text}")
        parts = text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else None

    # -- commands -------------------------------------------------------

    def read_u16(self, addr: int) -> int:
        resp = self._cmd(f"read_u16 0x{addr:X}")
        if resp is None:
            raise RuntimeError(f"Bridge returned no value for 'read_u16 0x{addr:X}'")
        return int(resp)

    def set_input(self, shoot: bool, peek: bool, aim_x: float = 0.5, aim_y: float = 0.5):
        # Apply the Guncon calibration exactly once, right before sending.
        cx, cy = apply_guncon_calibration(aim_x, aim_y)
        self._cmd(
            f"set_input {1 if shoot else 0} {1 if peek else 0} {cx:.4f} {cy:.4f}"
        )

    def step_frames(self, n: int = 1):
        self._cmd(f"step {int(n)}")

    def load_state(self, slot: int = 1):
        self._cmd(f"load {int(slot)}")

    def save_state(self, slot: int = 1):
        self._cmd(f"save {int(slot)}")

    def frame(self) -> int:
        resp = self._cmd("frame")
        if resp is None:
            raise RuntimeError("Bridge returned no value for 'frame'")
        return int(resp)

    def get_screenshot(self) -> np.ndarray:
        """Capture the current emulator frame as an HxWx3 uint8 RGB array.

        Sends the ``screenshot`` command, which the Lua bridge services by
        calling ``comm.socketServerScreenShot()``. BizHawk writes the frame as
        an uncompressed 32bpp BI_RGB BMP straight down the socket using the
        same length-prefix wire format as every other reply (verified against
        BizHawk 2.11.1 SocketServer.PrefixWithLength). Unlike every other
        command there is NO trailing ``OK`` -- the framed BMP IS the reply --
        so we read the payload as raw bytes via ``_recv_message_bytes`` and
        decode it through ``vision.decode_bmp`` here (never via _cmd, which
        would try to UTF-8 decode the binary payload).
        """
        if self.sock is None:
            raise RuntimeError("Bridge not connected. Call connect() first.")
        self._send("screenshot")
        bmp_bytes = self._recv_message_bytes()
        return decode_bmp(bmp_bytes)

    def hud(self, lines):
        safe = [str(s).replace("|", "/").replace("\n", " ") for s in lines]
        self._cmd("hud " + "|".join(safe))

    def hud_clear(self):
        self._cmd("hud_clear")

    def input_state(self) -> tuple[bool, bool, float, float]:
        """Return the bridge's latest applied input state.

        Payload shape: "OK <shoot01> <peek01> <aim_x> <aim_y>".
        Useful for debugging whether policies collapse to fixed patterns.
        """
        resp = self._cmd("input_state")
        if resp is None:
            raise RuntimeError("Bridge returned no value for 'input_state'")
        parts = resp.split()
        if len(parts) != 4:
            raise RuntimeError(f"Malformed input_state payload: {resp!r}")
        shoot = parts[0] == "1"
        peek = parts[1] == "1"
        aim_x = float(parts[2])
        aim_y = float(parts[3])
        return shoot, peek, aim_x, aim_y
