import json
import socket

import numpy as np


class FlappyEnv:
    backend = 'unity-v2'

    def __init__(self, host='127.0.0.1', port=9999, timeout=300):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self._buffer = b''
        try:
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.settimeout(timeout)
            self.server.bind((host, port))
            self.server.listen(1)
            print(f'Waiting for Unity on {host}:{port}...', flush=True)
            self.conn, _ = self.server.accept()
            self.conn.settimeout(timeout)
            self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except BaseException:
            self.close()
            raise

    def reset(self, seed=None):
        state, _, _, info = self._request({'command': 'reset', 'seed': int(seed or 0)})
        return state, info

    def step(self, action):
        if action not in (0, 1):
            raise ValueError('action must be 0 or 1')
        return self._request({'command': 'step', 'action': int(action)})

    def _request(self, command):
        try:
            self.conn.sendall((json.dumps(command) + '\n').encode('ascii'))
            while b'\n' not in self._buffer:
                chunk = self.conn.recv(4096)
                if not chunk:
                    raise ConnectionError('Unity disconnected')
                self._buffer += chunk
                if len(self._buffer) > 65536:
                    raise ValueError('Unity response exceeds protocol limit')
            line, self._buffer = self._buffer.split(b'\n', 1)
            result = json.loads(line)
            if result.get('error'):
                raise RuntimeError(result['error'])
            state = np.asarray(result['state'], dtype=np.float32)
            if result.get('protocol') != 2 or state.shape != (5,) or not np.isfinite(state).all():
                raise ValueError('Expected protocol 2 with five finite state values')
            return state, float(result['reward']), bool(result['done']), {
                'score': int(result['score']), 'backend': self.backend}
        except OSError as exc:
            self.close()
            raise ConnectionError('Unity connection failed; restart the environment') from exc

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        self.server.close()
