import csv
import os
import numpy as np
import torch
from environment import FlappyEnv
from dqn_agent import DQNAgent

env = FlappyEnv()
agent = DQNAgent()

CHECKPOINT = "checkpoints/dqn_new.pth"  # change this to whichever checkpoint you want
agent.epsilon = 1.0

if os.path.exists(CHECKPOINT):
    agent.policy_net.load_state_dict(torch.load(CHECKPOINT))
    agent.target_net.load_state_dict(torch.load(CHECKPOINT))
    print(f"Loaded checkpoint: {CHECKPOINT}")
else:
    print("No checkpoint found, starting fresh")

NUM_EPISODES = 5000
scores = []
best_avg_reward = float("-inf")

os.makedirs("checkpoints", exist_ok=True)

log_mode = "a" if os.path.exists(CHECKPOINT) and os.path.exists("training_log.csv") else "w"
log_file = open("training_log.csv", log_mode, newline="")
log_writer = csv.writer(log_file)

# only write header if starting fresh
if log_mode == "w":
    log_writer.writerow(["episode", "total_reward", "avg_reward_100", "epsilon"])

for episode in range(NUM_EPISODES):
    state, _, _ = env.step(0)
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        # if episode < 3:  # ADD THIS BLOCK
        #     print(f"State: {state} | Action: {action} | Reward: {reward} | Done: {done}")
        
        # print(f"Episode {episode:>4} | Reward: {total_reward:>7.2f} | Buffer: {len(agent.buffer)}")

        agent.store(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state
        total_reward += reward

        if done:
            break

    scores.append(total_reward)
    avg_reward = np.mean(scores[-100:])
    agent.decay_epsilon()

    log_writer.writerow([episode, round(total_reward, 2), round(avg_reward, 2), round(agent.epsilon, 4)])
    log_file.flush()

    if episode % 100 == 0:
        print(f"Episode {episode:>4} | Reward: {total_reward:>7.2f} | Avg(100): {avg_reward:>7.2f} | Epsilon: {agent.epsilon:.3f}")

    if episode % 500 == 0 and episode > 0:
        path = f"checkpoints/dqn_ep{episode}.pth"
        torch.save(agent.policy_net.state_dict(), path)
        print(f"Checkpoint saved: {path}")

    torch.save(agent.policy_net.state_dict(), "checkpoints/dqn_latest.pth")

    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        torch.save(agent.policy_net.state_dict(), "checkpoints/dqn_best.pth")
        print(f"New best model saved! Avg reward: {avg_reward:.2f}")

log_file.close()
print("Training complete!")