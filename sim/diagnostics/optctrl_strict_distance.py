"""strict × 거리 (V7): 60°에서 target을 멀리 두면 곧장(strict) 순항이 가능해지는가?

질문: optctrl_strict_reachable(V6)에서 60° 0.5m는 도달하나 게걸음(course_ss 0.56,
머리각 52~56°)으로만 가능 = strict 물리 천장. 단 0.5m는 짧아 "머리 틀어 순항 자세"
잡기 전에 끝남. 거리↑로 (a)머리 틀 공간↑ (b)transient 희석 되면 곧장 가능할 수도.
시간 제약(60s, 사실 #9)을 빼려고 episode를 거리 비례로 넉넉히 줘 순수 곧장능력만 측정.

판정: 거리↑로 course_ss가 0.70 넘고 머리각 40 아래 내려가면 → 거리로 곧장 가능(처방 후보).
      0.56 부근·머리각 50+ 유지 → 60° 곧장은 거리 무관 물리 천장 확정.

정책·학습 무관 read-only. V6의 rollout_strict/strict_cost/cem_strict 재사용.
사용: python3 diagnostics/optctrl_strict_distance.py
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
from optctrl_strict_reachable import rollout_strict, strict_cost, cem_strict
from fish_env import FishSwimEnv

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
PI = np.pi
SR = 0.08
DEG = 60
RADII = [0.5, 0.7, 1.0, 1.5]
SIDES = [("좌", +1), ("우", -1)]   # theta=PI+sgn*60°; 라벨은 실제 target y로 재확인
ITERS, POP = 18, 48   # course_ss 추세 + 도달 품질 균형. 멀리는 도달 자체가 어려워 예산 부족 시 미도달 인공물 주의.


def ep_sec(r):
    """거리 비례 episode (시간 제약 제거, 넉넉). 0.5→60, 1.0→80, 1.5→120s. 직진 1.5m는 ~15s라 충분."""
    return max(60.0, round(r * 80.0))


def main():
    np.random.seed(0)
    print(f"[V7 strict×거리] 60° 고정, 거리 sweep, 시간제약 제거(episode 거리비례).")
    print(f"  지표 = 정상상태 course_ss. 거리↑로 0.70 넘으면 곧장 가능, 0.56 유지면 물리 천장.\n", flush=True)
    print(f"{'r(m)':>5} {'side':<4} {'ep(s)':>5} {'best':>6} {'reach':>6} {'course':>7} "
          f"{'머리각':>6} {'course_ss':>9} {'strict':>7}")

    fig, axes = plt.subplots(len(RADII), len(SIDES), figsize=(5.2 * len(SIDES), 5.0 * len(RADII)))
    results = {}
    for ri, r in enumerate(RADII):
        t_sec = ep_sec(r)
        env = FishSwimEnv(episode_seconds=t_sec, success_radius=SR, action_history_n=20,
                          target_theta_offset_range=(PI / 12, PI / 12))
        n_steps = int(t_sec / env.dt)
        for ci, (slabel, sgn) in enumerate(SIDES):
            theta = PI + sgn * math.radians(DEG)
            # 시간제약 없음 = CEM 탐색 캡도 풀 episode (max_t=n_steps). r_target=r로 실제 거리 적용.
            _, bp = cem_strict(env, theta, n_steps, ITERS, POP, n_steps, r_target=r)
            best, reached, ca, ha, ss, css, xs, ys = rollout_strict(
                env, theta, bp, n_steps, record=True, r_target=r)
            ty = float(r * math.sin(theta))
            side_real = "우" if ty > 0 else "좌"
            results[(r, slabel)] = (best, reached, ca, ha, ss, css)
            print(f"{r:>5.1f} {slabel:<4} {t_sec:>5.0f} {best:>6.3f} {'Y' if reached else 'n':>6} "
                  f"{ca:>7.3f} {ha:>6.1f} {css:>9.3f} {'Y' if ss else 'n':>7}  (target {side_real})",
                  flush=True)

            ax = axes[ri][ci]
            ax.plot(xs, ys, '-', color='tab:green' if ss else ('tab:orange' if reached else 'tab:red'),
                    lw=1.5, alpha=0.85)
            ax.scatter(xs[::12], ys[::12], s=10, c=range(0, len(xs), 12), cmap='viridis', zorder=3)
            ax.plot(0, 0, 'k*', ms=14)
            tx = float(r * math.cos(theta))
            ax.add_patch(plt.Circle((tx, ty), SR, fc='none', ec='red', ls='--'))
            ax.plot(tx, ty, 'rx', ms=10)
            ax.set_title(f"r={r}m {slabel} strict={'Y' if ss else 'n'} "
                         f"course={ca:.2f}/ss={css:.2f} 머리각={ha:.0f}°")
            lim = r + 0.3
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect('equal'); ax.grid(alpha=0.3)
            ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        env.close()

    print("\n  판정 (거리↑로 60° 곧장 course_ss 개선되는가):")
    for slabel, _ in SIDES:
        line = "  ".join(f"r={r}: ss={results[(r, slabel)][5]:.2f} 머리각={results[(r, slabel)][3]:.0f}°"
                         for r in RADII)
        print(f"   {slabel}: {line}")
    base_ss = max(results[(0.5, "좌")][5], results[(0.5, "우")][5])
    far_ss = max(results[(RADII[-1], "좌")][5], results[(RADII[-1], "우")][5])
    if far_ss >= 0.70:
        print(f"\n  → 거리↑로 course_ss {base_ss:.2f}→{far_ss:.2f} (≥0.70): 곧장 가능 — 거리 처방 후보.")
    else:
        print(f"\n  → 거리↑로도 course_ss {base_ss:.2f}→{far_ss:.2f} (<0.70 유지): 60° 곧장 거리 무관 물리 천장.")

    fig.suptitle(f"strict×거리 {DEG}° (초록=strict/주황=도달만/빨강=미도달, 색=시간, x=target)", fontsize=13)
    out = "/home/yoo/RL_robot/images/optctrl_strict_distance.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"\n[plot] {out}")


if __name__ == "__main__":
    main()
