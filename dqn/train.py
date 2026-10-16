import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .dqn_agent import DQNAgent
from .evaluate import evaluate, make_env


def training_seed(base, episode):
    seed = base + episode
    if seed >= 10000:
        seed += 100
    if seed >= 20000:
        seed += 100
    if not 0 <= seed < 2**31:
        raise ValueError('Training seeds must fit a nonnegative signed 32-bit integer')
    return seed


def main():
    parser = argparse.ArgumentParser(description='Train Double DQN with independent evaluation')
    parser.add_argument('--backend', choices=['headless', 'unity'], default='headless')
    parser.add_argument('--steps', type=int, default=500000, help='Total environment step budget')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/dqn'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--eval-every', type=int, default=25000)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--epsilon-decay-steps', type=int, default=100000)
    parser.add_argument('--port', type=int, default=9999)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.eval_episodes < 1:
        parser.error('steps, eval-every and eval-episodes must be positive')
    torch.set_num_threads(1)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    agent = DQNAgent.load(args.resume) if args.resume else DQNAgent(
        seed=args.seed, epsilon_decay_steps=args.epsilon_decay_steps)
    env = make_env(args.backend, args.port)
    eval_env = make_env('headless') if args.backend == 'headless' else env
    if agent.metadata and agent.metadata.get('backend') != env.backend:
        raise ValueError('Resume backend differs from the checkpoint; evaluate transfer first')
    best_mean = agent.metadata.get('best_validation_mean', -1.)
    episode = agent.metadata.get('episode', 0)
    metadata = dict(agent.metadata)
    next_eval = (agent.environment_steps // args.eval_every + 1) * args.eval_every
    started = time.monotonic()
    recent = []
    log_path = args.run_dir / 'training.csv'
    with log_path.open('a' if args.resume else 'x', newline='') as log:
        writer = csv.writer(log)
        if log.tell() == 0:
            writer.writerow(['episode', 'steps', 'seed', 'pipes', 'reward', 'length', 'epsilon', 'loss'])
        try:
            while agent.environment_steps < args.steps:
                episode += 1
                seed = training_seed(agent.config['seed'], episode)
                state, info = env.reset(seed=seed)
                total, length, loss = 0., 0, None
                while True:
                    action = agent.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    agent.store(state, action, reward, next_state, done)
                    loss = agent.train_step()
                    state = next_state
                    total += reward
                    length += 1
                    if done or agent.environment_steps >= args.steps:
                        break
                recent.append(info['score'])
                writer.writerow([episode, agent.environment_steps, seed, info['score'],
                                 total, length, agent.epsilon, loss])
                log.flush()
                if episode % 100 == 0:
                    print(f'Episode {episode} | steps {agent.environment_steps} | '
                          f'pipes100 {np.mean(recent[-100:]):.2f} | '
                          f'best {max(recent)} | eps {agent.epsilon:.3f} | '
                          f'seconds {time.monotonic() - started:.0f}', flush=True)
                metadata = dict(episode=episode, backend=env.backend,
                                best_validation_mean=best_mean,
                                seed_scheme='base + episode, skipping 10000..10099 and 20000..20099')
                if agent.environment_steps >= next_eval:
                    report = evaluate(agent, eval_env, args.eval_episodes, 10000, max_steps=30000)
                    report['training_steps'] = agent.environment_steps
                    with (args.run_dir / 'validation.jsonl').open('a', encoding='utf-8') as output:
                        output.write(json.dumps(report) + '\n')
                    progress_path = args.run_dir / 'PROGRESSION.md'
                    if not progress_path.exists():
                        progress_path.write_text(
                            '# Validation progression\n\n'
                            f'Backend: {env.backend}. Frozen policy; seeds start at 10000.\n\n'
                            'Reward R0 is unchanged: +10 pass, -5 death, otherwise +0.1 survival, '
                            '-0.1 requested flap, +0.2/-0.1 gap alignment.\n\n'
                            '| Training steps | Mean pipes | Best pipes | Episodes | Truncated |\n'
                            '| --- | --- | --- | --- | --- |\n', encoding='utf-8')
                    with progress_path.open('a', encoding='utf-8') as progress:
                        progress.write(f'| {agent.environment_steps} | {report["mean_pipes"]:.2f} | '
                                       f'{report["best_pipes"]} | {report["episodes"]} | '
                                       f'{report["truncated_episodes"]} |\n')
                    print(f'VALIDATION steps {agent.environment_steps}: '
                          f'mean {report["mean_pipes"]:.2f}, best {report["best_pipes"]}, '
                          f'truncated {report["truncated_episodes"]}', flush=True)
                    if not report['truncated_episodes'] and report['mean_pipes'] > best_mean:
                        best_mean = report['mean_pipes']
                        metadata['best_validation_mean'] = best_mean
                        agent.save(args.run_dir / 'best.pth', metadata)
                    next_eval += args.eval_every
                if episode % 100 == 0:
                    agent.save(args.run_dir / 'latest.pth', metadata)
            agent.save(args.run_dir / 'latest.pth', metadata)
        finally:
            env.close()
            if eval_env is not env:
                eval_env.close()


if __name__ == '__main__':
    main()
