"""각 stage에서 deterministic rollout 경로(x, y) 시각화.

eval_stages.py와 동일 STAGES 정의 사용. 모델 1개를 6 stage에 각각 N ep 돌려서
시작점(0,0) → target 경로 trajectory를 plot. 2x3 subplot 1 PNG.

사용:
    python3 sim/diagnostics/plot_trajectories.py <model.zip> [--episodes 20]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from stable_baselines3 import SAC

# 한글 폰트 (Noto Sans CJK 설치 가정, 없으면 DejaVu fallback)
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
SIM_DIR = HERE.parent
sys.path.insert(0, str(SIM_DIR))
from eval_stages import STAGES, ACTION_HISTORY_N
from fish_env import FishSwimEnv


def rollout_stage(model, stage, n_episodes, seed_base):
    env = FishSwimEnv(
        episode_seconds=stage["ep_sec"],
        success_radius=stage["radius"],
        target_theta_range=stage["theta"],
        action_history_n=ACTION_HISTORY_N,
    )
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        target_xy = env._target_pos()[:2].copy()
        xs, ys = [], []
        xs.append(float(env._torso_pos()[0]))
        ys.append(float(env._torso_pos()[1]))
        terminated = truncated = False
        reached = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            pos = env._torso_pos()
            xs.append(float(pos[0]))
            ys.append(float(pos[1]))
            reached = bool(info.get("reached", False))
        episodes.append({
            "xs": np.array(xs),
            "ys": np.array(ys),
            "target": target_xy,
            "reached": reached,
        })
    env.close()
    return episodes


def plot_stage(ax, stage, episodes):
    n = len(episodes)
    n_reach = sum(1 for e in episodes if e["reached"])
    ax.set_title(f"{stage['tag']} — reach {n_reach}/{n} "
                 f"({100*n_reach/max(1,n):.0f}%)")

    # success_radius 원 (각 episode target 주위)
    seen = set()
    for e in episodes:
        key = (round(e["target"][0], 4), round(e["target"][1], 4))
        if key in seen:
            continue
        seen.add(key)
        ax.add_patch(Circle(e["target"], stage["radius"],
                            facecolor="none", edgecolor="0.7", lw=0.8, ls="--"))
        ax.plot(e["target"][0], e["target"][1], marker="x",
                color="0.4", ms=6, mew=1.2)

    # trajectories
    for e in episodes:
        color = "tab:green" if e["reached"] else "tab:red"
        alpha = 0.45 if e["reached"] else 0.85
        lw = 1.0 if e["reached"] else 1.5
        ax.plot(e["xs"], e["ys"], color=color, alpha=alpha, lw=lw)
        ax.plot(e["xs"][-1], e["ys"][-1], marker="o",
                color=color, ms=4, alpha=0.9, mec="none")

    # 시작점
    ax.plot(0, 0, marker="*", color="black", ms=12, mec="white", mew=0.6,
            zorder=5)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.axhline(0, color="0.85", lw=0.5, zorder=0)
    ax.axvline(0, color="0.85", lw=0.5, zorder=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=str, help="model.zip path")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None,
                   help="output directory (default: <model_dir>). "
                        "각 stage가 trajectories_<tag>.png로 저장됨.")
    args = p.parse_args()

    model_path = Path(args.model).resolve()

    out_dir = (Path(args.output).resolve() if args.output
               else model_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot] model={model_path}")
    print(f"[plot] {args.episodes} ep × {len(STAGES)} stages, seed_base={args.seed}")
    print(f"[plot] out_dir={out_dir}")
    model = SAC.load(str(model_path), device="cpu")

    subtitle = (f"world −x = 머리(전진). ★ 시작 (0,0), × target, "
                f"원 = success_radius. 초록=성공, 빨강=실패.")
    for stage in STAGES:
        print(f"  ... {stage['tag']}")
        episodes = rollout_stage(model, stage, args.episodes, args.seed)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_stage(ax, stage, episodes)
        fig.suptitle(f"Trajectories — {model_path.name} · {stage['tag']} "
                     f"({args.episodes} ep)\n{subtitle}",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = out_dir / f"trajectories_{stage['tag']}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"      saved {out.name}")


if __name__ == "__main__":
    main()
