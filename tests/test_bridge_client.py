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
    def __init__(self, host, port):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.commands = []
        self.error = None

    def run(self):
        try:
            time.sleep(0.1)
            with socket.create_connection((self.host, self.port), timeout=2.0) as sock:
                file = sock.makefile("rwb")
                while True:
                    raw = file.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8").strip()
                    self.commands.append(line)
                    if line.startswith("read_u16 "):
                        reply = "OK 4660\n"
                    elif line == "frame":
                        reply = "OK 99\n"
                    else:
                        reply = "OK\n"
                    file.write(reply.encode("utf-8"))
                    file.flush()
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

        self.assertEqual(client.read_u16(0x1234), 4660)
        client.set_input(True, False, aim_x=1.0, aim_y=0.25)
        self.assertEqual(client.frame(), 99)
        client.close()
        fake.join(timeout=2.0)

        self.assertIsNone(fake.error)
        self.assertEqual(fake.commands[0], "read_u16 0x1234")
        self.assertEqual(fake.commands[1], "set_input 1 0 0.9700 0.2500")
        self.assertEqual(fake.commands[2], "frame")

    def test_bridge_client_keeps_legacy_bias_fallback(self):
        host, port = "127.0.0.1", get_free_port()
        fake = FakeBizHawk(host, port)
        fake.start()

        client = BridgeClient(host, port, timeout=2.0)
        self.addCleanup(client.close)
        client.connect()
        client.set_input(False, True, aim_bias=1.0)
        client.close()
        fake.join(timeout=2.0)

        self.assertIsNone(fake.error)
        self.assertEqual(fake.commands, ["set_input 0 1 0.9700 0.5000"])


if __name__ == "__main__":
    unittest.main()
