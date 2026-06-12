"""strict reachability (V6): ±θ에서 졸업 게이트 success_strict가 최적제어로 달성 가능한가?

질문: 기존 optctrl_reachable는 목적=최소 도달거리뿐이라 게걸음이든 곧장이든 도달만
하면 cost 0. "±60°에서 곧장(strict: 도달 AND course≥0.70 AND 머리각≤40°)이 물리적으로
가능한가"는 검증된 적 없음. CEM 목적을 졸업 게이트와 동일한 strict로 바꿔 해 존재 여부 측정.

판정 (60° 결과):
  - strict 해 존재         → 학습 문제 (정책이 못 찾음).
  - 부재 + 정상상태 course≥0.70 → transient 천장 (게이트 metric 문제, 물리 아님).
  - 부재 + 정상상태 course 낮음 → 물리 천장 (s3c strict 미졸업 정상).

정책·학습 무관 read-only. optctrl_reachable의 set_target 재사용, strict cost용 rollout/cem 별도.
사용: python3 diagnostics/optctrl_strict_reachable.py
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
from fish_env import FishSwimEnv, IDX_X, IDX_Y

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
PI = np.pi
SR = 0.08
ANGLES = [15, 30, 45, 60]
SIDES = [("좌", +1), ("우", -1)]   # theta = PI + sgn*deg; 라벨은 실제 target y로 재확인

# strict cost 가중치 (plan): 도달이 1순위 게이트, 그 뒤 course/head 위반.
D_BASE = 2.5      # 미도달 바닥 > 도달 cost 상한(≈1.95) → CEM이 도달 포기 안 함
W_COURSE = 1.0
W_HEAD = 1.0


def rollout_strict(env, theta, params, n_steps, record=False, r_target=None, max_t=None):
    """optctrl_reachable.rollout 골격 + info에서 strict 메트릭 수집 + step별 course 직접 계산.
    반환: best, reached, course_avg, head_angle_avg, success_strict, course_ss [, xs, ys].
    course_avg/head_angle_avg/success_strict는 env info 그대로(fish_env 정의와 자동 일치).
    course_ss = 후반 50% step의 instantaneous course 평균 (transient vs 물리 천장 분리).
    max_t: CEM 탐색 비용 절감용 step 캡(0.5m는 빨리 도달 → 미도달 샘플만 조기 절단). 최종 평가는 None(풀 60s)."""
    env.reset(seed=O.SEED)
    O.set_target(env, theta, r_target)
    segs = params.reshape(O.K, 3)
    seg_len = max(1, n_steps // O.K)
    best = env._distance_to_target()
    reached = False
    course_avg, head_angle_avg, success_strict = 0.0, 90.0, False
    course_series = []   # step별 instantaneous course (v_norm>0.02일 때만, fish_env.py:419 일치)
    xs, ys = ([env._torso_pos()[0]], [env._torso_pos()[1]]) if record else (None, None)
    t = 0
    term = trunc = False
    while not (term or trunc):
        if max_t is not None and t >= max_t:
            break
        si = min(t // seg_len, O.K - 1)
        a = segs[si].astype(np.float32)
        _, _, term, trunc, info = env.step(a)
        d = env._distance_to_target()
        if d < best:
            best = d
        course_avg = info["course_avg"]
        head_angle_avg = info["head_angle_avg"]
        success_strict = bool(info["success_strict"])
        # instantaneous course = dot(v_world/|v|, rel/|rel|) (fish_env.py:424 동일식)
        v = np.array([float(env.data.qvel[IDX_X]), float(env.data.qvel[IDX_Y])])
        vn = float(np.linalg.norm(v))
        rel = env._target_pos()[:2] - env._torso_pos()[:2]
        rn = float(np.linalg.norm(rel))
        if vn > 0.02 and rn > 1e-6:
            course_series.append(float(np.dot(v / vn, rel / rn)))
        if record:
            p = env._torso_pos(); xs.append(p[0]); ys.append(p[1])
        if bool(info.get("reached", False)):
            reached = True
            break
        t += 1
    if course_series:
        half = len(course_series) // 2
        course_ss = float(np.mean(course_series[half:])) if course_series[half:] else float(np.mean(course_series))
    else:
        course_ss = 0.0
    if record:
        return best, reached, course_avg, head_angle_avg, success_strict, course_ss, np.array(xs), np.array(ys)
    return best, reached, course_avg, head_angle_avg, success_strict, course_ss


def strict_cost(best, reached, course_avg, head_angle_avg):
    if not reached:
        return D_BASE + best
    course_short = max(0.0, 0.70 - course_avg)
    head_over = max(0.0, head_angle_avg - 40.0)
    return W_COURSE * course_short + W_HEAD * (head_over / 40.0) + 0.05 * best


def cem_strict(env, theta, n_steps, iters, pop, max_t, r_target=None):
    dim = O.K * 3
    mean = np.zeros(dim)
    std = np.full(dim, 0.7)
    best_overall = (1e9, None)
    for _ in range(iters):
        samples = np.clip(mean[None] + std[None] * np.random.randn(pop, dim), -1.0, 1.0)
        costs = np.empty(pop)
        for i in range(pop):
            b, reached, ca, ha, _, _ = rollout_strict(env, theta, samples[i], n_steps,
                                                       max_t=max_t, r_target=r_target)
            costs[i] = strict_cost(b, reached, ca, ha)
        order = np.argsort(costs)
        elite = samples[order[:O.ELITE]]
        mean = elite.mean(0)
        std = elite.std(0) + 0.02
        if costs[order[0]] < best_overall[0]:
            best_overall = (float(costs[order[0]]), samples[order[0]].copy())
    return best_overall


def main():
    np.random.seed(0)
    env = FishSwimEnv(episode_seconds=60.0, success_radius=SR, action_history_n=20,
                      target_theta_offset_range=(PI / 12, PI / 12))
    n_steps = int(60.0 / env.dt)
    # CEM 탐색 rollout step 캡 (각도별): 작은 각은 빨리 도달해 25s 충분, 큰 각은 우회 경로 길어
    # 캡이 도달 자체를 막으면(60° 인공물) strict 판정 무효 → 45°=40s, 60°=풀 60s. 최종 평가는 항상 풀 60s.
    cap_by_deg = {15: int(25.0 / env.dt), 30: int(25.0 / env.dt),
                  45: int(40.0 / env.dt), 60: n_steps}
    print(f"[V6 strict reachability] CEM K={O.K}, strict cost (course≥0.70 & 머리각≤40° & 도달)")
    print(f"  목적 = 졸업 게이트 success_strict. 15/30°는 sanity(정책 v21 달성 → 해 나와야 정상).")
    print(f"  CEM 탐색 step캡(각도별)={cap_by_deg}, 최종 평가 풀 {n_steps}(60s).\n", flush=True)
    print(f"{'angle':>5} {'side':<4} {'best거리':>8} {'reach':>6} {'course':>7} {'머리각':>6} "
          f"{'course_ss':>9} {'strict':>7}")

    nrow, ncol = len(ANGLES), len(SIDES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.0 * nrow))
    results = {}
    for ri, deg in enumerate(ANGLES):
        # 60°만 예산 상향 (strict이 가장 어려운 핵심 각).
        iters, pop = (35, 64) if deg == 60 else (O.ITERS, O.POP)
        for ci, (slabel, sgn) in enumerate(SIDES):
            theta = PI + sgn * math.radians(deg)
            _, bp = cem_strict(env, theta, n_steps, iters, pop, cap_by_deg[deg])
            best, reached, ca, ha, ss, css, xs, ys = rollout_strict(
                env, theta, bp, n_steps, record=True)   # 최종 평가 = 풀 60s (max_t=None)
            ty = float(0.5 * math.sin(theta))   # r=0.5 고정
            side_real = "우" if ty > 0 else "좌"
            results[(deg, slabel)] = (best, reached, ca, ha, ss, css)
            print(f"{deg:>5} {slabel:<4} {best:>8.3f} {'Y' if reached else 'n':>6} "
                  f"{ca:>7.3f} {ha:>6.1f} {css:>9.3f} {'Y' if ss else 'n':>7}  (target {side_real})")

            ax = axes[ri][ci]
            ax.plot(xs, ys, '-', color='tab:green' if ss else ('tab:orange' if reached else 'tab:red'),
                    lw=1.5, alpha=0.85)
            ax.scatter(xs[::8], ys[::8], s=10, c=range(0, len(xs), 8), cmap='viridis', zorder=3)
            ax.plot(0, 0, 'k*', ms=14)
            tx = float(0.5 * math.cos(theta))
            ax.add_patch(plt.Circle((tx, ty), SR, fc='none', ec='red', ls='--'))
            ax.plot(tx, ty, 'rx', ms=10)
            ax.set_title(f"±{deg}° {slabel}  strict={'Y' if ss else 'n'}  "
                         f"course={ca:.2f}/ss={css:.2f} 머리각={ha:.0f}°")
            ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8)
            ax.set_aspect('equal'); ax.grid(alpha=0.3)
            ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    env.close()

    print("\n  판정 (60° = 핵심):")
    for slabel, _ in SIDES:
        b, reached, ca, ha, ss, css = results[(60, slabel)]
        if ss:
            verdict = "학습 문제 (최적제어로 곧장 도달 가능 → 정책이 못 찾음)"
        elif css >= 0.70:
            verdict = f"transient 천장 (정상상태 course {css:.2f}≥0.70, ep평균만 미달 → metric 문제)"
        else:
            verdict = f"물리 천장 (정상상태 course {css:.2f}도 낮음 → 게걸음만 가능, 미졸업 정상)"
        print(f"   60° {slabel}: strict={'Y' if ss else 'n'} course_avg={ca:.3f} course_ss={css:.3f} "
              f"머리각={ha:.1f}° → {verdict}")

    fig.suptitle("strict reachability (초록=strict달성/주황=도달만/빨강=미도달, 색=시간, x=target)",
                 fontsize=13)
    out = "/home/yoo/RL_robot/images/optctrl_strict_reachable.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"\n[plot] {out}")


if __name__ == "__main__":
    main()
