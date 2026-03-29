from dqn.environment import format_reset_command, format_step_command, parse_packet
from pathlib import Path


def test_parse_packet_returns_state_reward_done_and_pipes():
    packet = "0.1,-0.2,0.3,0.4,0.5,-0.6,1.0,0.03,0,12"
    state, reward, done, pipes = parse_packet(packet)

    assert state.shape == (7,)
    assert reward == 0.03
    assert done is False
    assert pipes == 12


def test_parse_packet_exposes_can_flap_in_last_observation_slot():
    state, reward, done, pipes = parse_packet("0,0,0,0,0,0,1,0,0,0")
    assert state[-1] == 1.0


def test_format_reset_command_includes_mode_and_seed():
    assert format_reset_command("eval", 42) == "RESET|eval|42\n"


def test_format_step_command_serializes_action():
    assert format_step_command(1) == "STEP|1\n"


def test_train_script_uses_reset_and_four_value_packets():
    source = Path("dqn/train.py").read_text(encoding="utf-8")

    assert "env.reset(" in source
    assert "state, _, _, _ = env.reset(" in source
    assert "next_state, reward, done, _ = env.step(action)" in source
