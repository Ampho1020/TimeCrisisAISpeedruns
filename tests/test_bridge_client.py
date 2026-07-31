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


if __name__ == "__main__":
    unittest.main()
