import csv
import os
import numpy as np
import torch

try:
    from dqn.environment import FlappyEnv
    from dqn.dqn_agent import DQNAgent
    from dqn.evaluation import run_evaluation, should_save_best
except ModuleNotFoundError:
    from environment import FlappyEnv
    from dqn_agent import DQNAgent
    from evaluation import run_evaluation, should_save_best

env = FlappyEnv()
agent = DQNAgent()

CHECKPOINT_BEST_EVAL = "checkpoints/dqn_best_eval_mean50.pth"
CHECKPOINT_LEGACY_BEST = "checkpoints/dqn_best.pth"
CHECKPOINT_LATEST = "checkpoints/dqn_latest.pth"
NUM_EPISODES = 5000
EVAL_INTERVAL = 100
EVAL_EPISODES = 50
best_eval_mean = float("-inf")

if os.path.exists(CHECKPOINT_BEST_EVAL):
    state_dict = torch.load(CHECKPOINT_BEST_EVAL)
    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {CHECKPOINT_BEST_EVAL}")
elif os.path.exists(CHECKPOINT_LEGACY_BEST):
    state_dict = torch.load(CHECKPOINT_LEGACY_BEST)
    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)
    print(f"Loaded legacy checkpoint: {CHECKPOINT_LEGACY_BEST}")
else:
    print("No checkpoint found, starting fresh")

scores = []

os.makedirs("checkpoints", exist_ok=True)

log_file = open("training_log.csv", "w", newline="")  # always overwrite
log_writer = csv.writer(log_file)
log_writer.writerow(
    [
        "episode",
        "train_total_reward",
        "train_pipes",
        "avg_train_reward_100",
        "eval_mean_pipes_50",
        "eval_median_pipes_50",
        "eval_std_pipes_50",
        "eval_early_deaths_50",
        "best_eval_mean_pipes_50",
        "epsilon",
    ]
)

for episode in range(1, NUM_EPISODES + 1):
    state, _, _, _ = env.reset(mode="train", seed=episode)
    total_reward = 0
    pipes_passed = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, pipes_passed = env.step(action)

        agent.store(state, action, reward, next_state, done)
        if len(agent.buffer) >= agent.min_buffer_size:
            agent.train_step()

        state = next_state
        total_reward += reward

        if done:
            break

    scores.append(total_reward)
    avg_reward = np.mean(scores[-100:])
    eval_metrics = None

    if episode % EVAL_INTERVAL == 0:
        eval_metrics = run_evaluation(
            env,
            agent,
            num_episodes=EVAL_EPISODES,
            base_seed=10000 + episode,
        )
        if should_save_best(best_eval_mean, eval_metrics["mean_pipes"]):
            best_eval_mean = eval_metrics["mean_pipes"]
            torch.save(agent.policy_net.state_dict(), CHECKPOINT_BEST_EVAL)
            print(
                f"New best eval model saved: {CHECKPOINT_BEST_EVAL} "
                f"(mean pipes {best_eval_mean:.2f})"
            )

    log_writer.writerow(
        [
            episode,
            round(total_reward, 4),
            pipes_passed,
            round(avg_reward, 4),
            round(eval_metrics["mean_pipes"], 4) if eval_metrics else "",
            round(eval_metrics["median_pipes"], 4) if eval_metrics else "",
            round(eval_metrics["std_pipes"], 4) if eval_metrics else "",
            eval_metrics["early_deaths"] if eval_metrics else "",
            round(best_eval_mean, 4) if best_eval_mean != float("-inf") else "",
            round(agent.epsilon, 6),
        ]
    )
    log_file.flush()

    print(
        f"Episode {episode:>4} | Train Reward: {total_reward:>8.2f} "
        f"| Train Pipes: {pipes_passed:>4} | Epsilon: {agent.epsilon:.3f}"
    )

    if episode % 100 == 0:
        summary = (
            f"Episode {episode:>4} | Avg Train Reward(100): {avg_reward:>8.2f} "
            f"| Epsilon: {agent.epsilon:.3f}"
        )
        if eval_metrics:
            summary += (
                f" | Eval Mean(50): {eval_metrics['mean_pipes']:.2f}"
                f" | Eval Median(50): {eval_metrics['median_pipes']:.2f}"
                f" | Eval Std(50): {eval_metrics['std_pipes']:.2f}"
                f" | Eval EarlyDeaths(50): {eval_metrics['early_deaths']}"
            )
        print(summary)

    if episode % 500 == 0 and episode > 0:
        path = f"checkpoints/dqn_ep{episode}.pth"
        torch.save(agent.policy_net.state_dict(), path)
        print(f"Checkpoint saved: {path}")

    torch.save(agent.policy_net.state_dict(), CHECKPOINT_LATEST)

log_file.close()
print("Training complete!")
