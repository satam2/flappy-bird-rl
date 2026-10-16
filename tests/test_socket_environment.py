import json
import socket
import threading

import pytest


def test_socket_reset_and_step_receive_fragmented_json_and_authoritative_score():
    from dqn.environment import FlappyEnv
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    listener.close()
    commands, errors = [], []

    def unity_peer():
        try:
            import time
            for _ in range(100):
                try:
                    peer = socket.create_connection(('127.0.0.1', port), timeout=2)
                    break
                except ConnectionRefusedError:
                    time.sleep(.01)
            with peer, peer.makefile('rb') as stream:
                for score in (0, 1):
                    commands.append(json.loads(stream.readline()))
                    message = json.dumps(dict(state=[0, .2, .4, -.1, 0], reward=10,
                                              done=False, score=score, protocol=2)) + '\n'
                    peer.sendall(message[:7].encode())
                    peer.sendall(message[7:].encode())
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=unity_peer)
    thread.start()
    env = FlappyEnv(port=port, timeout=3)
    try:
        state, info = env.reset(seed=8)
        assert len(state) == 5 and info['score'] == 0
        assert env.step(1)[3]['score'] == 1
        with pytest.raises(ConnectionError):
            env.step(0)
    finally:
        env.close()
        thread.join(timeout=3)
    assert not errors
    assert commands == [{'command': 'reset', 'seed': 8}, {'command': 'step', 'action': 1}]
