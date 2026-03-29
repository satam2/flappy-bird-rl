import socket
import numpy as np


def parse_packet(packet: str):
    values = list(map(float, packet.split(",")))
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

    def step(self, action):
        # send action
        self.conn.sendall(f"{action}\n".encode())

        # receive one complete line from Unity
        data = self._recv_line()
        parts = list(map(float, data.split(",")))
        state  = np.array(parts[:4], dtype=np.float32)
        reward = parts[4]
        done   = bool(parts[5])
        return state, reward, done

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
