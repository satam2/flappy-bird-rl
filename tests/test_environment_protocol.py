from dqn.environment import parse_packet


def test_parse_packet_returns_state_reward_done_and_pipes():
    packet = "0.1,-0.2,0.3,0.4,0.5,-0.6,1.0,0.03,0,12"
    state, reward, done, pipes = parse_packet(packet)

    assert state.shape == (7,)
    assert reward == 0.03
    assert done is False
    assert pipes == 12
