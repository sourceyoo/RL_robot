"""strict 60° 천장 — 통제 재측정 (V8): multi-seed CEM으로 σ를 직접 측정.

배경: V6/V7(optctrl_strict_reachable/distance)은 episode 길이·예산이 제각각이고 seed
효과를 안 봐서, 같은 60° 셀의 머리각이 38~98°로 출렁였다(단일 noisy draw). 결론 불가.

이 진단: episode 정책·예산을 하나로 고정하고 seed만 바꿔, "최적제어가 strict 해를 얼마나
안정적으로 찾는가"의 σ를 측정한다. rollout은 env.reset(seed=O.SEED) 고정 → 물리는 결정적,
유일한 변동원은 CEM 샘플링(np.random). 따라서 σ = 해 탐색 신뢰도(물리 σ 아님).

sanity(30°,0.5m): v21이 strict 달성한 각·거리 → CEM도 3 seed 다 strict=Y여야 정상.
  안정적이면 CEM이 충분히 강하다는 증거 → 60° 실패를 물리 신호로 해석 가능.
  30°도 간헐이면 CEM 약함 → 60° 판정 보류(예산↑ 재실행).

정책·학습 무관 read-only. rollout_strict/strict_cost/cem_strict는 V6에서 import 재사용.
사용: python3 diagnostics/optctrl_strict_controlled.py [--smoke]
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
from optctrl_strict_reachable import rollout_strict, cem_strict
from fish_env import FishSwimEnv

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
PI = np.pi
SR = 0.08

SMOKE = "--smoke" in sys.argv

# 셀 = (deg, r). (30,0.5)=sanity, (60,1.0)/(60,1.5)=핵심. 가까운 0.5/0.7m 60°는 미도달 인공물이라 제외.
CELLS = [(60, 1.5)] if SMOKE else [(30, 0.5), (60, 1.0), (60, 1.5)]
SIDES = [("좌", +1)] if SMOKE else [("좌", +1), ("우", -1)]   # theta=PI+sgn*deg; 라벨은 실제 target y로 재확인
SEEDS = [0] if SMOKE else [0, 1, 2]
ITERS, POP = (5, 16) if SMOKE else (30, 80)
CAP_SEC = 90.0   # CEM 탐색 rollout step 캡(런타임 절감). 도달을 막으면 인공물 → 넉넉히. 최종 평가는 full ep.


def ep_sec(r):
    """거리 비례 episode 한 가지로 통일 (시간 제약 제거). 0.5→60, 1.0→120, 1.5→180s."""
    return float(round(r * 120))


def main():
    print(f"[V8 통제 재측정] CELLS={CELLS} SIDES={[s[0] for s in SIDES]} SEEDS={SEEDS} "
          f"ITERS={ITERS} POP={POP} CAP={CAP_SEC:.0f}s{'  (SMOKE)' if SMOKE else ''}")
    print(f"  σ = CEM 해 탐색 신뢰도(물리 결정적). 30° sanity 안정 strict=Y여야 CEM 신뢰.\n", flush=True)
    print(f"{'deg':>4} {'r(m)':>5} {'side':<4} {'seed':>4} {'ep(s)':>5} {'best':>6} {'reach':>6} "
          f"{'course':>7} {'머리각':>6} {'course_ss':>9} {'strict':>7}")

    # (deg,r,side) -> list over seeds: dict with metrics + trajectory
    results = {}
    for (deg, r) in CELLS:
        t_sec = ep_sec(r)
        env = FishSwimEnv(episode_seconds=t_sec, success_radius=SR, action_history_n=20,
                          target_theta_offset_range=(PI / 12, PI / 12))
        n_steps = int(t_sec / env.dt)
        cap = min(n_steps, int(CAP_SEC / env.dt))
        for (slabel, sgn) in SIDES:
            theta = PI + sgn * math.radians(deg)
            ty = float(r * math.sin(theta))
            side_real = "우" if ty > 0 else "좌"
            key = (deg, r, slabel)
            results[key] = {"side_real": side_real, "theta": theta, "r": r, "seeds": []}
            for s in SEEDS:
                np.random.seed(s)
                _, bp = cem_strict(env, theta, n_steps, ITERS, POP, cap, r_target=r)
                best, reached, ca, ha, ss, css, xs, ys = rollout_strict(
                    env, theta, bp, n_steps, record=True, r_target=r)   # 최종 평가 = full ep
                results[key]["seeds"].append(
                    {"seed": s, "best": best, "reached": reached, "ca": ca, "ha": ha,
                     "ss": ss, "css": css, "xs": xs, "ys": ys})
                print(f"{deg:>4} {r:>5.1f} {slabel:<4} {s:>4} {t_sec:>5.0f} {best:>6.3f} "
                      f"{'Y' if reached else 'n':>6} {ca:>7.3f} {ha:>6.1f} {css:>9.3f} "
                      f"{'Y' if ss else 'n':>7}  (target {side_real})", flush=True)
        env.close()

    # 종합: (deg,r,side) 별 strict k/3, course/머리각/course_ss mean±σ
    print(f"\n  [종합] strict k/{len(SEEDS)}, mean±σ (course / 머리각 / course_ss):")
    print(f"{'deg':>4} {'r(m)':>5} {'side':<4} {'strict':>7} {'course':>14} {'머리각':>14} {'course_ss':>14}")
    for key in results:
        deg, r, slabel = key
        S = results[key]["seeds"]
        k = sum(1 for x in S if x["ss"])
        ca = np.array([x["ca"] for x in S]); ha = np.array([x["ha"] for x in S]); css = np.array([x["css"] for x in S])
        print(f"{deg:>4} {r:>5.1f} {slabel:<4} {k}/{len(SEEDS):>5} "
              f"{ca.mean():>6.2f}±{ca.std():>5.2f} {ha.mean():>6.1f}±{ha.std():>5.1f} "
              f"{css.mean():>6.2f}±{css.std():>5.2f}")

    # 판정 (60° 종합)
    print("\n  [판정] (60° 핵심):")
    sixty = [k for k in results if k[0] == 60]
    if not sixty:
        print("   60° 셀 없음(SMOKE).")
    else:
        # sanity 점검
        thirty = [k for k in results if k[0] == 30]
        sanity_ok = all(all(x["ss"] for x in results[k]["seeds"]) for k in thirty) if thirty else None
        if thirty:
            sk = {k: sum(1 for x in results[k]["seeds"] if x["ss"]) for k in thirty}
            print(f"   sanity 30°: strict {sk} (전 seed Y여야 CEM 신뢰) → {'OK' if sanity_ok else '미달 → 60° 판정 보류, 예산↑ 재실행'}")
        # 60° 좌우 종합
        ks = {k: sum(1 for x in results[k]["seeds"] if x["ss"]) for k in sixty}
        min_k = min(ks.values()); max_k = max(ks.values())
        all_css = np.concatenate([[x["css"] for x in results[k]["seeds"]] for k in sixty])
        all_ha = np.concatenate([[x["ha"] for x in results[k]["seeds"]] for k in sixty])
        print(f"   60° strict k: {ks}")
        if max_k == 0:
            if all_css.mean() >= 0.70:
                print(f"   → 전 seed strict=n + course_ss {all_css.mean():.2f}≥0.70, 머리각 {all_ha.mean():.0f}°"
                      f": 곡률 한계 물리 천장(진행 곧장 가능하나 머리 못 틈).")
            else:
                print(f"   → 전 seed strict=n + course_ss {all_css.mean():.2f}<0.70: 물리 천장(게걸음만).")
        elif min_k >= 2:
            print(f"   → 좌우 모두 ≥2/3 seed strict=Y: 거리로 가능 = 학습 문제(정책이 못 찾음).")
        else:
            print(f"   → strict 간헐(min {min_k}/{len(SEEDS)}): 해 존재하나 좁음 = 학습 매우 어려움.")
        print("   ⚠ 숫자 전 trajectory(plot) 확인 필수.")

    # plot: 60° 셀 대표 seed (strict=Y 우선, 없으면 머리각 최소 = strict 근접)
    plot_keys = [k for k in results if k[0] == 60]
    if plot_keys:
        ncol = len(SIDES) if not SMOKE else 1
        nrow = max(1, len(plot_keys) // max(1, ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.0 * nrow), squeeze=False)
        flat = axes.flatten()
        for ax, key in zip(flat, plot_keys):
            deg, r, slabel = key
            S = results[key]["seeds"]
            rep = next((x for x in S if x["ss"]), min(S, key=lambda x: x["ha"]))
            xs, ys = rep["xs"], rep["ys"]
            color = 'tab:green' if rep["ss"] else ('tab:orange' if rep["reached"] else 'tab:red')
            ax.plot(xs, ys, '-', color=color, lw=1.5, alpha=0.85)
            ax.scatter(xs[::12], ys[::12], s=10, c=range(0, len(xs), 12), cmap='viridis', zorder=3)
            ax.plot(0, 0, 'k*', ms=14)
            theta = results[key]["theta"]
            tx, ty = float(r * math.cos(theta)), float(r * math.sin(theta))
            ax.add_patch(plt.Circle((tx, ty), SR, fc='none', ec='red', ls='--'))
            ax.plot(tx, ty, 'rx', ms=10)
            ax.set_title(f"{deg}° {slabel} seed{rep['seed']} strict={'Y' if rep['ss'] else 'n'} "
                         f"course={rep['ca']:.2f}/ss={rep['css']:.2f} 머리각={rep['ha']:.0f}°")
            lim = r + 0.3
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect('equal'); ax.grid(alpha=0.3)
            ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        fig.suptitle("strict 60° 통제 재측정 (대표 seed; 초록=strict/주황=도달만/빨강=미도달, 색=시간, x=target)",
                     fontsize=13)
        out = "/home/yoo/RL_robot/images/optctrl_strict_controlled.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        print(f"\n[plot] {out}")


if __name__ == "__main__":
    main()
