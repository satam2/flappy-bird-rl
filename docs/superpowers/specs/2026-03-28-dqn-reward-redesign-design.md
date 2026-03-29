# DQN And Reward Redesign For Reliable 150-Pipe Evaluation

## Summary

This design upgrades the Flappy Bird RL training stack so the saved model is judged by the right target: a mean score of at least 150 pipes over 50 evaluation episodes. The redesign keeps the current Unity scene, components, prefabs, and sprites intact, but allows script-level changes in Unity and Python where they improve training stability and policy quality.

The main direction is:

- improve the observation space so the agent sees the control problem directly
- simplify and stabilize the reward signal so pipe passing dominates
- replace vanilla DQN with Double DQN and a more stable training loop
- separate noisy training reward from evaluation-based checkpointing

## Goals

- Train a policy that reaches a mean score of at least 150 pipes across 50 evaluation episodes.
- Preserve the current game presentation, assets, and component structure.
- Allow Unity script changes where they improve observability, reward quality, reset behavior, and evaluation consistency.
- Keep the initial implementation focused on the highest-leverage changes before introducing optional complexity.

## Non-Goals

- Changing sprites, prefabs, scene art, UI art, or the visible style of the game.
- Changing the benchmark through easier permanent gameplay physics or looser evaluation criteria.
- Introducing advanced RL methods beyond what is needed to materially improve the current DQN stack.

## Current System

The current project uses:

- a Unity environment that sends 4 normalized values: bird Y, bird Y velocity, next pipe X distance, and next gap center Y
- a reward with death penalty, pipe pass reward, per-step living reward, unconditional flap penalty, and a threshold-based gap bonus/penalty
- a Python trainer using a replay buffer, a policy network, a target network, and epsilon-greedy exploration
- checkpointing driven by rolling training reward rather than a dedicated evaluation loop

This setup is functional, but it has two planning-level weaknesses:

1. The agent sees too little state for reliable long-horizon control.
2. The training objective and checkpoint selection do not match the actual success criterion.

## Design Overview

The redesign has four coordinated parts:

1. Expand the observation space with directly useful control features.
2. Replace static heuristic-heavy reward shaping with transition-based shaping that remains secondary to passing pipes and avoiding death.
3. Upgrade the agent from vanilla DQN to Double DQN with more stable optimization defaults.
4. Add explicit evaluation episodes and save the best checkpoint based on mean score over 50 evaluation episodes.

## Architecture

### Unity Side

Unity remains the source of truth for environment state, reward computation, and episode lifecycle. The main Unity changes are script changes only:

- `FlappyEnvironment.cs` becomes the RL environment manager
- `PipeSpawnHandler.cs` becomes responsible for robust access to the next and following pipes
- `BirdController.cs` exposes action-readiness state needed for observation and control

The scene, public components, and art assets remain unchanged.

### Python Side

Python remains responsible for:

- action selection
- replay storage
- value learning
- exploration schedule
- evaluation loop
- checkpoint selection

The Python trainer should treat Unity as an environment that returns stable observations, rewards, and done signals.

## Observation Design

The observation space should be expanded from 4 values to a compact set that directly represents the control problem.

Recommended observation vector:

- normalized bird Y position
- normalized bird Y velocity
- normalized X distance to the next pipe
- normalized Y delta to the next gap center
- normalized X distance to the following pipe
- normalized Y delta to the following gap center
- flap readiness, either as a binary `can_flap` signal or normalized cooldown remaining

Rationale:

- `dy_to_gap` is a more useful control feature than sending absolute bird Y and gap Y alone.
- The following pipe gives the policy short-horizon planning context that matters for long streaks.
- Flap readiness matters because the agent's action space is constrained by cooldown timing.

Optional future observation additions, only if needed later:

- time since last flap
- a compact safe-corridor estimate around the next gap

## Reward Design

The reward should be dominated by real game progress and failure, with shaping only used to improve credit assignment.

### Primary Reward Terms

- death: `-1.5` to `-2.0`
- pass pipe: `+1.0`

These are the dominant events and should remain more important than dense shaping accumulated between pipes.

### Dense Shaping

Dense shaping should reward whether the transition improved the bird's relation to the next target gap.

Recommended shaping signal:

- compute `previous_abs_dy_to_gap`
- compute `current_abs_dy_to_gap`
- shaping is a small value based on the improvement from previous to current
- clamp shaping to a small interval such as `[-0.03, +0.03]`

This produces a smoother learning signal than fixed threshold rules and helps the agent learn corrective actions earlier.

### Action Penalty

The unconditional flap penalty should be removed or reduced to a negligible value. If any action penalty remains, it should only target clearly wasteful flaps, such as flapping while already rising sharply in a poor region of the state space.

### Living Reward

The living reward should be removed or reduced to near zero, such as `0.0` to `+0.01`. Large living rewards risk teaching the agent to value survival artifacts more than clean pipe traversal.

## Agent Design

The agent should move from vanilla DQN to Double DQN as the default upgrade path.

### Required Changes

- use Double DQN target computation
- replace MSE loss with Huber loss
- add gradient clipping
- increase network capacity moderately
- update target parameters with soft updates or less aggressive hard-copy timing

Recommended network shape:

- input dimension equal to the new observation vector
- hidden layers around `128 -> 128 -> 64`
- output dimension of 2 actions

### Exploration

Exploration should decay by environment steps rather than only per episode. This makes the exploration schedule easier to reason about and less sensitive to episode length changes as the agent improves.

The existing strong bias against flapping during exploration should be reconsidered after the new state and reward system are in place. It may still be useful, but it should not be treated as a core learning mechanism.

### Optional Later Upgrades

These are deferred unless the first pass is not enough:

- dueling network heads
- prioritized replay
- n-step returns

The first implementation should avoid stacking too many changes at once.

## Training And Evaluation Design

### Training Loop

Training episodes should continue to use epsilon-greedy exploration and replay-based learning, but checkpoint selection should no longer depend on rolling training reward alone.

Training flow:

1. reset environment
2. collect transitions with exploration enabled
3. train after minimum replay fill threshold is reached
4. periodically run evaluation episodes with exploration disabled
5. save checkpoints based on evaluation metrics

### Evaluation Loop

Evaluation should run with `epsilon = 0.0` and no learning updates.

Required evaluation metrics:

- mean pipes over 50 episodes
- median pipes over 50 episodes
- standard deviation over 50 episodes
- early-death count

Primary success rule:

- the main checkpoint of interest is the one with evaluation mean of at least 150 over 50 episodes

### Checkpointing

Keep separate checkpoints for:

- latest training state
- best evaluation mean over 50 episodes
- optional diagnostic checkpoints at coarse intervals

The best model should be chosen using the same criterion the user actually cares about.

## Unity Script Responsibilities

### `FlappyEnvironment.cs`

This script should:

- gather the full observation vector
- compute reward from state transition
- manage episode reset lifecycle
- maintain previous-step values needed for shaping
- serialize observation, reward, and done values to Python
- support evaluation mode if deterministic or seeded evaluation is added

### `PipeSpawnHandler.cs`

This script should:

- expose the next and following pipes reliably
- expose gap-center information in a stable way
- support reset-safe pipe bookkeeping
- optionally support controlled randomness or seeding later

### `BirdController.cs`

This script should:

- expose flap-readiness state
- keep physics and action execution local to the bird
- avoid taking on reward or trainer responsibilities

## Error Handling And Robustness

The implementation should reduce failure modes that would make long training runs unreliable.

Required robustness expectations:

- socket disconnects should fail clearly instead of silently corrupting training
- environment reset state should not leak data from the previous episode
- pipe tracking should handle destroyed objects safely
- evaluation mode should not accidentally train or explore

## Testing And Verification

The implementation plan should include verification at three levels:

### Unity/Environment Verification

- confirm the environment sends the new observation size consistently
- confirm reward values follow the intended hierarchy in a small set of controlled scenarios
- confirm resets clear prior-episode shaping state

### Python/Agent Verification

- confirm the new observation dimension matches the agent input dimension
- confirm Double DQN target computation is used
- confirm evaluation episodes run with exploration disabled and no gradient updates

### Outcome Verification

- compare training curves before and after the redesign
- compare evaluation mean and median over 50 episodes
- inspect whether catastrophic early deaths decrease

## Implementation Order

Recommended order:

1. expand Unity observation output and cleanup state bookkeeping
2. update Python environment parsing and agent input size
3. implement Double DQN, Huber loss, and gradient clipping
4. replace reward shaping with transition-based shaping
5. add evaluation loop and evaluation-based checkpoint selection
6. add optional curriculum or later upgrades only if the first pass underperforms

## Risks

- Too much shaping could still overpower the true objective if not tightly clamped.
- Adding too many RL upgrades at once could make regression diagnosis harder.
- Evaluation without controlled randomness may remain noisy enough to slow checkpoint selection.
- If the following-pipe features are implemented incorrectly, the agent may learn from inconsistent targets.

## Open Decisions For Planning

These should be resolved in the implementation plan, not left open during coding:

- exact numeric values for death penalty and shaping clamp
- whether evaluation should use seeded randomness
- whether the first implementation should include dueling heads or remain plain Double DQN
- minimum replay size before training begins

The default recommendation is to start with plain Double DQN, no prioritized replay, no dueling heads, and deterministic evaluation only if measurement noise proves problematic.
