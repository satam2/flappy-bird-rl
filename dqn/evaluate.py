import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from .dqn_agent import DQNAgent


def make_env(backend, port=9999):
    if backend == 'headless':
        from .headless import HeadlessFlappyEnv
        return HeadlessFlappyEnv()
    from .environment import FlappyEnv
    return FlappyEnv(port=port)


def evaluate(agent, env, episodes=100, seed_start=20000, max_steps=100000):
    if episodes < 1 or max_steps < 1:
        raise ValueError('episodes and max_steps must be positive')
    scores, rewards, lengths, truncated = [], [], [], 0
    for seed in range(seed_start, seed_start + episodes):
        state, info = env.reset(seed=seed)
        total = 0.
        done = False
        for length in range(1, max_steps + 1):
            state, reward, done, info = env.step(agent.select_action(state, explore=False))
            total += reward
            if done:
                break
        truncated += int(not done)
        scores.append(int(info['score']))
        rewards.append(total)
        lengths.append(length)
    return {
        'backend': env.backend, 'unity_verified': env.backend == 'unity-v2',
        'episodes': episodes, 'seeds': list(range(seed_start, seed_start + episodes)),
        'scores': scores, 'episode_rewards': rewards, 'episode_lengths': lengths,
        'mean_pipes': float(np.mean(scores)), 'median_pipes': float(np.median(scores)),
        'best_pipes': max(scores), 'min_pipes': min(scores),
        'std_pipes': float(np.std(scores)), 'truncated_episodes': truncated,
        'max_steps_per_episode': max_steps, 'exploration': False, 'training_updates': 0,
        'target_met': bool(episodes >= 100 and truncated == 0 and
                           np.mean(scores) >= 45 and max(scores) >= 74),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate a frozen DQN using actual pipe scores')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--backend', choices=['headless', 'unity'], default='headless')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=20000)
    parser.add_argument('--max-steps', type=int, default=100000)
    parser.add_argument('--output', type=Path, default=Path('runs/evaluation.json'))
    parser.add_argument('--port', type=int, default=9999)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    agent = DQNAgent.load(io.BytesIO(checkpoint_bytes))
    env = make_env(args.backend, args.port)
    try:
        report = evaluate(agent, env, args.episodes, args.seed_start, args.max_steps)
    finally:
        env.close()
    report['checkpoint_sha256'] = checkpoint_hash
    report['training_metadata'] = agent.metadata
    if args.backend == 'headless':
        from .headless import RULES
        report['rules'] = RULES
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in
                      ('backend', 'episodes', 'mean_pipes', 'best_pipes',
                       'truncated_episodes', 'target_met')}, indent=2))
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
