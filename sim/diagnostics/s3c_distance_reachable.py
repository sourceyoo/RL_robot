"""Part A (V5): s3c(±60°) 도달성이 target 거리에 따라 개선되는가?

가설: 거리를 늘리면 같은 ±60°라도 곡률 반경 여유가 생겨 도달 가능 영역이 넓어진다
(거리↑ = s3c 난이도↓). CEM 최적제어로 거리별 좌/우 best 도달거리·reach 측정.
정책·학습 무관 read-only. optctrl_reachable의 cem/rollout/set_target(r_target) 재사용.

사용: python3 diagnostics/s3c_distance_reachable.py
판정: 거리↑에서 우측 best거리가 success_radius(0.08) 아래로 내려가면 거리 확대 학습 후보.
"""
import math, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import optctrl_reachable as O
from fish_env import FishSwimEnv

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
PI = np.pi
SR = 0.08
DEG = 60
RADII = [0.5, 0.7, 1.0, 1.5]
SIDES = [("좌(-y)", +1), ("우(+y)", -1)]  # theta = PI + sgn*60°; 라벨은 실제 target y로 재확인


def main():
    np.random.seed(0)
    env = FishSwimEnv(episode_seconds=60.0, success_radius=SR, action_history_n=20,
                      target_theta_offset_range=(PI / 12, PI / 12))
    n_steps = int(60.0 / env.dt)
    print(f"[V5 s3c 거리별 reachability] CEM K={O.K}, pop={O.POP}, iter={O.ITERS}, ±{DEG}°")
    print(f"  도달 = best거리 < {SR}. target y>0=우 / y<0=좌 (v22_subcheck 규약)\n")
    print(f"{'거리(m)':>7} {'side':<7} {'best거리':>8} {'reach':>6} {'도달step':>8} {'시간(s)':>7}")

    nrow, ncol = len(RADII), len(SIDES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.0 * nrow))
    results = {}
    for ri, r in enumerate(RADII):
        for ci, (slabel, sgn) in enumerate(SIDES):
            theta = PI + sgn * math.radians(DEG)
            bd, bp = O.cem(env, theta, n_steps, r_target=r)
            best, reached, xs, ys = O.rollout(env, theta, bp, n_steps, record=True, r_target=r)
            # 실제 target y로 좌/우 재확인
            ty = float(r * math.sin(theta))
            side_real = "우" if ty > 0 else "좌"
            results[(r, slabel)] = (best, reached, len(xs))
            print(f"{r:>7.2f} {slabel:<7} {best:>8.3f} {'Y' if reached else 'n':>6} "
                  f"{len(xs):>8d} {len(xs) * env.dt:>7.1f}  (target {side_real})")

            ax = axes[ri][ci]
            ax.plot(xs, ys, '-', color='tab:green' if reached else 'tab:red', lw=1.5, alpha=0.8)
            ax.scatter(xs[::8], ys[::8], s=10, c=range(0, len(xs), 8), cmap='viridis', zorder=3)
            ax.plot(0, 0, 'k*', ms=14)
            tx = float(r * math.cos(theta))
            ax.add_patch(plt.Circle((tx, ty), SR, fc='none', ec='red', ls='--'))
            ax.plot(tx, ty, 'rx', ms=10)
            ax.set_title(f"r={r}m {slabel}  도달={'Y' if reached else 'n'} best={best:.3f}")
            lim = r + 0.3
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect('equal'); ax.grid(alpha=0.3)
    env.close()

    print("\n  판정 (거리↑로 우측 도달 개선되는가):")
    for r in RADII:
        bl = results[(r, "좌(-y)")][0]
        br = results[(r, "우(+y)")][0]
        print(f"   r={r}m: 좌 {bl:.3f}{'(도달)' if bl < SR else ''} / "
              f"우 {br:.3f}{'(도달)' if br < SR else ''}")

    fig.suptitle(f"s3c ±{DEG}° 거리별 CEM reachability (별=시작, x=target, 색=시간, 초록=도달/빨강=실패)",
                 fontsize=13)
    out = "/home/yoo/RL_robot/images/v5_s3c_distance_reachable.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"\n[plot] {out}")


if __name__ == "__main__":
    main()
