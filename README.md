# Flappy Bird RL Project

This project trains a Deep Q-Network (DQN) to play a Unity-based Flappy Bird clone. Unity runs the game simulation, Python runs the reinforcement learning loop, and the two processes communicate over a local TCP socket on `127.0.0.1:9999`.

The main idea is:

- Unity exposes the game as an environment.
- Python chooses an action each step: `0` = do nothing, `1` = flap.
- Unity returns the next state, a reward, and whether the episode ended.
- Python stores transitions in replay memory and trains a DQN with PyTorch.

## What The Code Is Doing

The training loop lives in [`dqn/train.py`](./dqn/train.py).

For each episode, the trainer:

1. waits for the Unity scene to connect,
2. sends an action to Unity,
3. receives a state vector, reward, and done flag,
4. stores the transition in replay memory,
5. performs a DQN training step,
6. logs progress and saves checkpoints.

### State Space

Unity sends a 4-value normalized state from [`flappy_bird/Assets/Scripts/FlappyEnvironment.cs`](./flappy_bird/Assets/Scripts/FlappyEnvironment.cs):

- bird Y position
- bird vertical velocity
- horizontal distance to the next pipe
- Y position of the next pipe gap

### Action Space

- `0`: no flap
- `1`: flap

### Reward Shaping

The current reward function is:

- `+10.0` for passing a pipe
- `-5.0` for dying
- `+0.1` for surviving a step
- `-0.1` for flapping
- `-0.1` when the bird is far from the next gap
- `+0.2` when the bird stays close to the next gap

### Training Behavior

The current DQN implementation in [`dqn/dqn_agent.py`](./dqn/dqn_agent.py) uses:

- a 2-layer MLP with hidden size `128`
- replay buffer capacity `50000`
- batch size `64`
- discount factor `0.99`
- Adam optimizer with learning rate `3e-4`
- target network sync every `100` training steps
- epsilon-greedy exploration with decay

Checkpoints are written by [`dqn/train.py`](./dqn/train.py) to:

- `dqn/checkpoints/dqn_latest.pth`
- `dqn/checkpoints/dqn_best.pth`
- `dqn/checkpoints/dqn_ep500.pth`, `dqn_ep1000.pth`, etc.

Training metrics are also written to `dqn/training_log.csv`.

## Repo Layout

```text
flappy_bird_rl_proj/
|-- dqn/
|   |-- train.py
|   |-- environment.py
|   |-- dqn_agent.py
|   `-- replay_buffer.py
|-- flappy_bird/
|   |-- Assets/
|   |-- Packages/
|   `-- ProjectSettings/
`-- README.md
```

## Prerequisites

### Unity

- Unity editor version `6000.3.2f1`

This is the version recorded in [`flappy_bird/ProjectSettings/ProjectVersion.txt`](./flappy_bird/ProjectSettings/ProjectVersion.txt).

### Python

This repo does not currently include a pinned `requirements.txt`, so set up a Python environment with at least:

- `numpy`
- `torch`

Example:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy torch
```

## How To Run It

### 1. Open the Unity project

Open the `flappy_bird` folder in Unity Hub using Unity `6000.3.2f1`.

After the project loads, open:

- `Assets/Scenes/GameScene.unity`

### 2. Start the Python trainer

From the repo root:

```powershell
cd dqn
python train.py
```

At this point the trainer starts a TCP server and waits for Unity to connect. You should see:

```text
Waiting for Unity to connect...
```

### 3. Start the Unity simulation

With `GameScene` open, press Play in the Unity editor.

When Play mode starts:

- Unity connects to the Python process on `127.0.0.1:9999`
- the menu UI is hidden
- episodes start automatically
- the episode counter updates inside the scene

Once Unity connects, Python begins training immediately.

## Expected Output

While training runs:

- Unity shows the bird playing repeated episodes
- the Python console prints per-episode rewards
- `training_log.csv` is overwritten and updated live
- model checkpoints are saved into `dqn/checkpoints/`

If `dqn/checkpoints/dqn_best.pth` already exists, `train.py` loads it before training. If it does not exist, training starts from scratch.

## Resuming From A Checkpoint

Checkpoint loading is currently controlled directly in [`dqn/train.py`](./dqn/train.py):

- `CHECKPOINT` sets which model file to load
- `agent.epsilon` sets the starting exploration rate
- `best_avg_reward` sets the baseline used to decide when to overwrite `dqn_best.pth`

If you want to resume from a different checkpoint, edit those values before running the script.

## Notes

- The Python trainer should usually be started before pressing Play in Unity, because Unity retries the socket connection until the Python server is available.
- The bird can still be flapped manually with mouse input because that behavior is present in [`flappy_bird/Assets/Scripts/BirdController.cs`](./flappy_bird/Assets/Scripts/BirdController.cs), but the intended workflow here is automated RL training.
- This repo is focused on training inside the Unity editor. There is not currently a separate evaluation or inference script.
