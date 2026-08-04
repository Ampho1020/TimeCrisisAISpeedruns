import socket
import threading
import time
import unittest

from bridge_client import BridgeClient, apply_guncon_calibration


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
