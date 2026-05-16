"""모델 한 개를 모든 stage 분포에서 평가 — catastrophic forgetting 측정.

사용:
    python3 sim/eval_stages.py <model.zip> [--episodes 100]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent))
from fish_env import FishSwimEnv

PI = math.pi
ACTION_HISTORY_N = 20

STAGES = [
    # ep_sec은 curriculum.py STAGES와 sync (m4 환경: s1/s2 20s, s3a~d 60s).
    {"tag": "s1_forward", "theta": (PI, PI),                       "radius": 0.08, "ep_sec": 20.0},
    {"tag": "s2_anchor",  "theta": (PI, PI),                       "radius": 0.04, "ep_sec": 20.0},
    {"tag": "s3a_arc15",  "theta": (PI - PI/12, PI + PI/12),       "radius": 0.08, "ep_sec": 60.0},
    {"tag": "s3b_arc30",  "theta": (PI - PI/6,  PI + PI/6),        "radius": 0.08, "ep_sec": 60.0},
    {"tag": "s3c_arc60",  "theta": (PI - PI/3,  PI + PI/3),        "radius": 0.08, "ep_sec": 60.0},
    {"tag": "s3d_arc90",  "theta": (PI/2,        3*PI/2),          "radius": 0.08, "ep_sec": 60.0},
]


def eval_stage(model, stage, n_episodes, seed):
    env = FishSwimEnv(
        episode_seconds=stage["ep_sec"],
        success_radius=stage["radius"],
        target_theta_range=stage["theta"],
        action_history_n=ACTION_HISTORY_N,
    )
    reaches = 0
    aligns = []
    final_dists = []
    ep_seconds = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        ep_align_sum = 0.0
        ep_step = 0
        last_dist = None
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            ep_align_sum += float(info.get("align", 0.0))
            ep_step += 1
            last_dist = float(info.get("distance", 0.0))
        if info.get("reached", False):
            reaches += 1
        aligns.append(ep_align_sum / max(1, ep_step))
        final_dists.append(last_dist if last_dist is not None else 0.0)
        ep_seconds.append(ep_step * env.dt)
    env.close()
    return {
        "reach_rate": reaches / n_episodes,
        "avg_align":  float(np.mean(aligns)),
        "final_dist": float(np.mean(final_dists)),
        "ep_seconds": float(np.mean(ep_seconds)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=str, help="model.zip path")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model = SAC.load(args.model, device="cpu")
    print(f"\n[eval] model={args.model}, n_ep={args.episodes}, seed_base={args.seed}\n")
    print(f"{'stage':<12} {'reach':>7} {'align':>7} {'fdist':>7} {'ep_s':>6}")
    print("-" * 44)
    for stage in STAGES:
        r = eval_stage(model, stage, args.episodes, args.seed)
        print(f"{stage['tag']:<12} {r['reach_rate']*100:>6.1f}% "
              f"{r['avg_align']:>+7.3f} {r['final_dist']:>7.3f} "
              f"{r['ep_seconds']:>6.2f}")


if __name__ == "__main__":
    main()
