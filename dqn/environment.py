import socket
import numpy as np


def format_reset_command(mode: str, seed: int) -> str:
    return f"RESET|{mode}|{seed}\n"


def format_step_command(action: int) -> str:
    return f"STEP|{action}\n"


def parse_packet(packet: str):
    values = list(map(float, packet.strip().split(",")))
    state = np.array(values[:7], dtype=np.float32)
    reward = values[7]
    done = bool(int(values[8]))
    pipes = int(values[9])
    return state, reward, done, pipes


class FlappyEnv:
    def __init__(self, host="127.0.0.1", port=9999):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        print("Waiting for Unity to connect...")
        self.conn, _ = self.server.accept()
        print("Connected!")
        self._buffer = ""  # persistent buffer

    def reset(self, mode="train", seed=0):
        self.conn.sendall(format_reset_command(mode, seed).encode())
        packet = self._recv_line()
        return parse_packet(packet)

    def step(self, action):
        self.conn.sendall(format_step_command(action).encode())
        packet = self._recv_line()
        return parse_packet(packet)

    def _recv_line(self):
        # keep reading until we have a complete line
        while "\n" not in self._buffer:
            chunk = self.conn.recv(256).decode()
            if not chunk:
                raise ConnectionError("Unity disconnected")
            self._buffer += chunk

        # split off the first complete line
        line, self._buffer = self._buffer.split("\n", 1)
        return line.strip()
