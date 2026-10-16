# Flappy Bird RL

A Unity Flappy Bird project with a PyTorch Double DQN trainer, explicit pipe
scores, reproducible evaluation, and a Box2D headless training backend.

**Headless scores are proxy results. They do not establish Unity performance.**
The headless backend copies the scene geometry and settings, but Unity uses
its own Physics2D version and random-number generator. Run Unity evaluation
to measure transfer. No pretrained weights were present in the original repo.

The included checkpoint is `runs/reference42/best.pth`, selected after 75,107
training steps. On 100 held-out headless episodes it scored **96.7 pipes on
average, best 377**, with no truncated episodes. Validation's best was 378.
All scores, seeds and the model hash are in `runs/reference42/evaluation.json`.

## Setup

Tested on Windows, Python 3.12, CPU. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

## Train Without Unity

```powershell
python -m dqn.train --backend headless --steps 500000 --seed 42 --run-dir runs/new42
```

The step budget is total steps, including any restored checkpoint. Training
logs every episode to `training.csv` without confusing reward and pipe score.
`best.pth` is chosen by a separate greedy validation mean; `latest.pth` saves
training progress. Each checkpoint includes weights, target network,
optimizer, replay memory, RNG state, exploration and training configuration.
Only load checkpoints you trust.

```powershell
python -m dqn.train --resume runs/new42/latest.pth --steps 1000000 --run-dir runs/new42
```

Resume starts a fresh episode using the next deterministic training seed.
It does not restore an in-flight game. Use a new run directory for a fresh
experiment; an existing training log is never silently overwritten.

## Evaluate a Frozen Policy

```powershell
python -m dqn.evaluate runs/reference42/best.pth --backend headless --episodes 100 --seed-start 20000 --output runs/reference42/recheck.json
```

Evaluation disables exploration and training. JSON retains every seed, pipe
score, episode length and reward, plus checkpoint SHA-256 and backend.
Training uses `seed + episode`, skipping the reserved ranges 10000..10099
and 20000..20099. Validation starts at 10000; final evaluation uses
20000..20099. Custom evaluation ranges should also be disjoint from training.

The target gate requires at least 100 complete episodes, mean pipes >=45,
and best pipes >=74. Higher scores are accepted. Episodes are never stopped
because they reach a target score. The emergency step limit defaults to
100000; any episode reaching it is reported as truncated and disqualifies
the run from passing the target gate.

## Open the Unity Project

The repository contains the full project, not just Python:

1. Install Unity **6000.3.2f1**, the version recorded in
   `flappy_bird/ProjectSettings/ProjectVersion.txt`.
2. In Unity Hub, choose **Add project from disk** and select `flappy_bird`.
3. Open `Assets/Scenes/GameScene.unity`.
4. Start Python evaluation below, then press **Play** in Unity.

```powershell
python -m dqn.evaluate runs/reference42/best.pth --backend unity --episodes 100 --seed-start 20000 --output runs/reference42/unity-evaluation.json
```

To train directly in Unity:

```powershell
python -m dqn.train --backend unity --steps 500000 --run-dir runs/unity42
```

The trainer listens on `127.0.0.1:9999`; Unity connects to it. Restart Play
mode and Python together after a disconnect. The default 300-second socket
timeout fails explicitly rather than injecting fabricated actions.

## Simulation Contract

Protocol 2 sends newline-delimited JSON. `reset` sets a seed and returns the
initial observation. Each `step` applies action 0/1, advances exactly 0.02
seconds of physics, and returns the resulting observation, reward, terminal
flag and authoritative score. No physics advances while Python is busy.

Observations are bird Y / 5, vertical velocity / 10, next pipe distance / 8,
gap Y / 5, and flap cooldown / 0.15. The cooldown makes flap eligibility
observable. Four-input legacy weights and the original CSV socket protocol
are incompatible and require retraining/updating both processes together.

Pipe speed (4.79), pipe geometry, gap offsets (-2..2), gravity scale (3.01),
flap force (12), upward velocity cap (10), and cooldown (0.15s) are retained
from the original scene and scripts. Pipe movement, spawning, cooldown and
physics share one simulation clock in RL mode. Manual mouse input is disabled
during agent control. Human play still uses the existing frame updates.

Rewards retain +10 on pipe sensor entry, -5 on death, +0.1 survival, -0.1
requested flap, and +0.2/-0.1 gap alignment. Pipe counts are reported
independently and each pipe can score at most once. A sensor entry on a
fatal tick does not award a pipe in either backend.

## Changes and Evidence

`runs/reference42/training.csv` and `validation.jsonl` retain the corrected
training curve and model selection results. `evaluation.json` is the separate
100-episode test. `runs/seed42` retains only exploratory logs; its proxy physics
ordering was corrected and that run was discarded as final evidence.

The DQN now uses Double DQN targets, Huber loss, gradient clipping, replay
warmup and step-based exploration. These address value overestimation,
outlier updates, correlated early samples and inadequate fresh-run exploration.
An independent review agent inspected the changes; 16 Python tests passed.
Unity compilation, runtime behavior and policy transfer remain unverified.

The deterministic physics reference applies the flap impulse, integrates
physics, then rotates/caps velocity. Spawning uses an idealized 51-tick cadence;
the original frame-dependent float timer can take 52 ticks. This normalizes
timing and does not establish exact equivalence to the original rendered game.

Detailed local reward progression, review reports and planning documents are
intentionally ignored by Git. Future validations also write a local
`PROGRESSION.md` in their run directory.
