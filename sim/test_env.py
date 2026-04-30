"""FishSwimEnv smoke test — 환경이 Gymnasium API를 정상 따르는지 확인."""

import numpy as np
from fish_env import FishSwimEnv

env = FishSwimEnv()
print(f"obs_space: {env.observation_space}")
print(f"act_space: {env.action_space}")
print(f"max_steps/episode: {env.max_steps}")

# random rollout
obs, info = env.reset(seed=0)
print(f"reset obs shape: {obs.shape}, init distance: {info if info else env._prev_distance:.4f}")

total_reward = 0.0
for t in range(env.max_steps):
    action = env.action_space.sample()
    obs, r, terminated, truncated, info = env.step(action)
    total_reward += r
    if terminated or truncated:
        break

print(f"random episode: steps={t+1}  total_reward={total_reward:.3f}  "
      f"final_distance={info['distance']:.4f}  reached={info['reached']}")

# sanity check on observation
print(f"obs sample: {obs}")
assert not np.any(np.isnan(obs)), "obs has NaN!"
print("=== OK ===")
env.close()
