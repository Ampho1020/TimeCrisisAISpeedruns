import socket
import struct
import threading
import time
import unittest

import numpy as np

from bridge_client import BridgeClient, apply_guncon_calibration


def _make_bmp(width: int, height: int, fill_rgb=(11, 22, 33)) -> bytes:
    """Produce a valid uncompressed 32bpp BI_RGB BMP filled with a solid color.

    Height is stored as POSITIVE (bottom-up per BMP spec, matching vision.py's
    decode_bmp expectations). Layout mirrors what BizHawk's
    ``comm.socketServerScreenShot`` produces (32bpp XRGB, uncompressed), so
    FakeBizHawk can stand in for the real emulator on the receiving end.
    """
    row_bytes = width * 4  # already a multiple of 4 for 32bpp
    pixel_bytes = bytearray(row_bytes * height)
    r, g, b = fill_rgb
    for i in range(0, len(pixel_bytes), 4):
        pixel_bytes[i:i + 4] = bytes((b, g, r, 0))  # BGRA layout on disk
    pixel_offset = 54
    file_size = pixel_offset + len(pixel_bytes)
    header = bytearray(54)
    header[0:2] = b"BM"
    struct.pack_into("<I", header, 2, file_size)
    struct.pack_into("<I", header, 10, pixel_offset)
    struct.pack_into("<I", header, 14, 40)  # DIB header size
    struct.pack_into("<i", header, 18, width)
    struct.pack_into("<i", header, 22, height)
    struct.pack_into("<H", header, 26, 1)   # planes
    struct.pack_into("<H", header, 28, 32)  # bit count
    struct.pack_into("<I", header, 30, 0)   # BI_RGB
    struct.pack_into("<I", header, 34, len(pixel_bytes))
    return bytes(header) + bytes(pixel_bytes)


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeBizHawk(threading.Thread):
    """Stand-in for BizHawk's comm.* socket server.

    Mirrors the real wire protocol so the tests exercise the same code paths as
    production:
      * every message (both directions) is length-prefixed as "{len} {payload}"
        (BizHawk's format since 2.6.2),
      * "READY" is announced once, unprompted, the way the Lua script does,
      * the handshake "hello" messages are answered with "ERR unknown_cmd",
        exactly like the Lua bridge -- which is what makes the leftover replies
        pile up and reproduces the resync bug the client must drain past.
    """

    def __init__(self, host, port):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.commands = []
        self.error = None
        # Optional canned screenshot response for the "screenshot" command.
        # If set, servicing that command sends this BMP payload framed with
        # the standard "{N} <bytes>" length prefix -- exactly mirroring
        # BizHawk's real comm.socketServerScreenShot() behaviour (no trailing
        # OK reply, the framed BMP IS the reply).
        self.screenshot_bytes: bytes | None = None

    @staticmethod
    def _frame(payload):
        data = payload.encode("utf-8")
        return f"{len(data)} ".encode("utf-8") + data

    @staticmethod
    def _recv_message(sock, buf):
        while b" " not in buf:
            chunk = sock.recv(4096)
            if chunk == b"":
                return None, buf
            buf += chunk
        length_str, _, rest = buf.partition(b" ")
        n = int(length_str)
        buf = rest
        while len(buf) < n:
            chunk = sock.recv(4096)
            if chunk == b"":
                return None, buf
            buf += chunk
        return buf[:n].decode("utf-8"), buf[n:]

    def run(self):
        try:
            time.sleep(0.1)
            with socket.create_connection((self.host, self.port), timeout=2.0) as sock:
                sock.settimeout(2.0)
                buf = b""
                # Announce readiness once, unprompted (mirrors the Lua script).
                sock.sendall(self._frame("READY"))
                while True:
                    line, buf = self._recv_message(sock, buf)
                    if line is None:
                        break
                    if line == "hello":
                        # Same as the Lua bridge: handshake spam is "unknown".
                        sock.sendall(self._frame("ERR unknown_cmd"))
                        continue
                    self.commands.append(line)
                    if line.startswith("read_u16 "):
                        reply = "OK 4660"
                    elif line == "frame":
                        reply = "OK 99"
                    elif line == "screenshot":
                        # Match the real bridge: send the framed BMP as the
                        # SOLE reply (no trailing OK). If no canned bytes are
                        # set, fall back to a tiny default so any test that
                        # accidentally issues "screenshot" without configuring
                        # a payload fails loudly on decode rather than hanging.
                        payload = self.screenshot_bytes or b""
                        sock.sendall(f"{len(payload)} ".encode("utf-8") + payload)
                        continue
                    else:
                        reply = "OK"
                    sock.sendall(self._frame(reply))
        except Exception as exc:  # pragma: no cover - surfaced by assertions
            self.error = exc


class BridgeClientTests(unittest.TestCase):
    def test_apply_guncon_calibration_scales_x_once(self):
        x, y = apply_guncon_calibration(1.0, 0.25)
        self.assertAlmostEqual(x, 0.97)
        self.assertAlmostEqual(y, 0.25)

    def test_bridge_client_accepts_connection_and_sends_explicit_aim(self):
        host, port = "127.0.0.1", get_free_port()
        fake = FakeBizHawk(host, port)
        fake.start()

        client = BridgeClient(host, port, timeout=2.0)
        self.addCleanup(client.close)
        client.connect()
        # Drop the handshake/resync traffic so only real commands are asserted.
        fake.commands.clear()

        self.assertEqual(client.read_u16(0x1234), 4660)
        client.set_input(True, False, aim_x=1.0, aim_y=0.25)
        self.assertEqual(client.frame(), 99)
        client.close()
        fake.join(timeout=2.0)

        self.assertIsNone(fake.error)
        self.assertEqual(fake.commands[0], "read_u16 0x1234")
        self.assertEqual(fake.commands[1], "set_input 1 0 0.9700 0.2500")
        self.assertEqual(fake.commands[2], "frame")

    def test_get_screenshot_reads_framed_bmp_and_decodes_to_rgb(self):
        """BridgeClient.get_screenshot() must consume the length-prefixed BMP
        exactly (no trailing OK reply, no double-framing) and hand it off to
        vision.decode_bmp to produce a valid HxWx3 uint8 RGB array. This is
        the Phase 1 round-trip gate for the vision plan -- if this passes,
        the same wire code will handle the real BizHawk BMP."""
        host, port = "127.0.0.1", get_free_port()
        fake = FakeBizHawk(host, port)
        fake.screenshot_bytes = _make_bmp(8, 4, fill_rgb=(11, 22, 33))
        fake.start()

        client = BridgeClient(host, port, timeout=2.0)
        self.addCleanup(client.close)
        client.connect()
        fake.commands.clear()

        frame = client.get_screenshot()
        self.assertEqual(frame.shape, (4, 8, 3))
        self.assertEqual(frame.dtype, np.uint8)
        # Every pixel must decode to the (11, 22, 33) RGB fill regardless of
        # bottom-up BMP row order -- catches any accidental BGR mix-up or
        # payload-alignment bug in _recv_message_bytes.
        self.assertTrue(np.all(frame[:, :, 0] == 11))
        self.assertTrue(np.all(frame[:, :, 1] == 22))
        self.assertTrue(np.all(frame[:, :, 2] == 33))

        # After the screenshot, ordinary text commands must still work: the
        # bytes primitive must have left the recv buffer empty and not
        # de-synced the framing for the next text-mode reply.
        self.assertEqual(client.frame(), 99)
        client.close()
        fake.join(timeout=2.0)
        self.assertIsNone(fake.error)
        self.assertEqual(fake.commands, ["screenshot", "frame"])


class PeekHoldRewardTest(unittest.TestCase):
    """Lock in that holding peek is rewarded and spamming never is."""

    def test_spamming_earns_nothing(self):
        from env_timecrisis import peek_hold_reward

        # Toggle every tick: every run is length 1, so nothing accrues.
        spam = [i % 2 == 0 for i in range(30)]
        self.assertEqual(peek_hold_reward(spam, traverse_ticks=3, reward=4.0), 0.0)

    def test_partial_hold_earns_dense_gradient(self):
        from env_timecrisis import peek_hold_reward

        # A single length-2 hold earns one step of reward -- more than a tap,
        # less than a full commit. This slope is what lets ES climb.
        tap = [True, False]
        hold2 = [True, True, False]
        hold3 = [True, True, True, False]
        r_tap = peek_hold_reward(tap, traverse_ticks=3, reward=4.0)
        r_hold2 = peek_hold_reward(hold2, traverse_ticks=3, reward=4.0)
        r_hold3 = peek_hold_reward(hold3, traverse_ticks=3, reward=4.0)
        self.assertEqual(r_tap, 0.0)
        self.assertEqual(r_hold2, 4.0)
        self.assertEqual(r_hold3, 8.0)
        self.assertLess(r_tap, r_hold2)
        self.assertLess(r_hold2, r_hold3)

    def test_holding_past_traverse_is_capped(self):
        from env_timecrisis import peek_hold_reward

        # Holding forever is capped at (traverse - 1) * reward: no camping bonus.
        self.assertEqual(
            peek_hold_reward([True] * 30, traverse_ticks=3, reward=4.0), 8.0
        )

    def test_holding_strictly_beats_spamming(self):
        from env_timecrisis import peek_hold_reward

        spam = [i % 2 == 0 for i in range(30)]
        cycles = ([True] * 3 + [False]) * 7  # deliberate hold/release cycles
        self.assertGreater(
            peek_hold_reward(cycles, traverse_ticks=3, reward=4.0),
            peek_hold_reward(spam, traverse_ticks=3, reward=4.0),
        )


if __name__ == "__main__":
    unittest.main()
