from pathlib import Path

import numpy as np


def summarize_scores(scores, early_deaths):
    return {
        "mean_pipes": float(np.mean(scores)),
        "median_pipes": float(np.median(scores)),
        "std_pipes": float(np.std(scores)),
        "early_deaths": early_deaths,
    }


def should_save_best(current_best, candidate_mean):
    return candidate_mean > current_best


def load_best_eval_mean(path):
    metric_path = Path(path)
    if not metric_path.exists():
        return float("-inf")

    try:
        metric_value = float(metric_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return float("-inf")

    if not np.isfinite(metric_value):
        return float("-inf")

    return metric_value


def save_best_eval_mean(path, metric_value):
    metric_path = Path(path)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(str(float(metric_value)), encoding="utf-8")


def run_evaluation(env, agent, num_episodes=50, base_seed=10000):
    scores = []
    early_deaths = 0

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    try:
        for episode_idx in range(num_episodes):
            state, _, done, pipes = env.reset(mode="eval", seed=base_seed + episode_idx)

            while not done:
                action = agent.select_action(state)
                state, _, done, pipes = env.step(action)

            scores.append(pipes)
            if pipes == 0:
                early_deaths += 1
    finally:
        agent.epsilon = original_epsilon

    return summarize_scores(scores, early_deaths)
