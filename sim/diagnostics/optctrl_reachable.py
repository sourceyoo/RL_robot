"""V1: 독립 최적제어 reachability (정책·학습 무관, read-only).

질문: SAC 정책이 아니라 직접 최적화한 시변 제어로 각 stage target에 도달하는
action sequence가 존재하는가? 존재하면 "정책이 못 찾은 학습 문제", 없으면 "물리 한계".

방법: CEM. action을 K-segment piecewise-constant (freq,amp,offset)로 파라미터화,
목적 = ep 최소 도달거리. s3a/s3b는 도달해야 sanity OK (정책 100%). s3c/s3d가 핵심.
사용: python3 diagnostics/optctrl_reachable.py
"""
import math, sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv, IDX_YAW

PI = np.pi
SEED = 4000
K = 8           # segment 수
POP = 56        # population
ELITE = 8
ITERS = 22
SR = 0.08
# 각 stage 대표각 (annular 중앙). 좌(+)·우(-) 둘 다 (fin 비대칭 있으므로).
ANGLES = {"s3a(15°)": 15, "s3b(30°)": 30, "s3c(60°)": 60, "s3d(90°)": 90}


def set_target(env, theta):
    """timevarying_reachable.set_target 복제: target 위치 + turn state 재계산."""
    r = float(np.linalg.norm(env._target_pos()[:2]))
    gid = env._target_geom_id
    env.model.geom_pos[gid] = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
    mujoco.mj_forward(env.model, env.data)
    yaw0 = float(env.data.qpos[IDX_YAW])
    hd0 = np.array([-np.cos(yaw0), np.sin(yaw0)])
    rel0 = env._target_pos()[:2] - env._torso_pos()[:2]
    n = float(np.linalg.norm(rel0))
    env._yaw0 = yaw0
    env._turn_sign = -float(np.sign(hd0[0] * rel0[1] - hd0[1] * rel0[0])) if n > 1e-6 else 0.0
    env._target_offset = float(np.arccos(np.clip(float(np.dot(hd0, rel0 / n)), -1.0, 1.0))) if n > 1e-6 else 0.0
    env._prev_distance = env._distance_to_target()
    env._prev_pos = env._torso_pos()[:2].copy()


def rollout(env, theta, params, n_steps, record=False):
    env.reset(seed=SEED)
    set_target(env, theta)
    segs = params.reshape(K, 3)
    seg_len = max(1, n_steps // K)
    best = env._distance_to_target()
    reached = False
    xs, ys = ([env._torso_pos()[0]], [env._torso_pos()[1]]) if record else (None, None)
    t = 0
    term = trunc = False
    while not (term or trunc):
        si = min(t // seg_len, K - 1)
        a = segs[si].astype(np.float32)
        _, _, term, trunc, info = env.step(a)
        d = env._distance_to_target()
        if d < best:
            best = d
        if record:
            p = env._torso_pos(); xs.append(p[0]); ys.append(p[1])
        if bool(info.get("reached", False)):
            reached = True
            break
        t += 1
    if record:
        return best, reached, np.array(xs), np.array(ys)
    return best, reached


def cem(env, theta, n_steps):
    dim = K * 3
    mean = np.zeros(dim)
    std = np.full(dim, 0.7)
    best_overall = (1e9, None)
    for it in range(ITERS):
        samples = np.clip(mean[None] + std[None] * np.random.randn(POP, dim), -1.0, 1.0)
        dists = np.empty(POP)
        for i in range(POP):
            d, reached = rollout(env, theta, samples[i], n_steps)
            dists[i] = 0.0 if reached else d
        order = np.argsort(dists)
        elite = samples[order[:ELITE]]
        mean = elite.mean(0)
        std = elite.std(0) + 0.02
        if dists[order[0]] < best_overall[0]:
            best_overall = (float(dists[order[0]]), samples[order[0]].copy())
        if best_overall[0] < SR:  # 이미 도달
            break
    return best_overall


def main():
    np.random.seed(0)
    env = FishSwimEnv(episode_seconds=60.0, success_radius=SR, action_history_n=20,
                      target_theta_offset_range=(PI / 12, PI / 12))
    n_steps = int(60.0 / env.dt)
    print(f"[V1 최적제어 reachability] CEM K={K} seg, pop={POP}, elite={ELITE}, iter={ITERS}")
    print(f"  목적=최소 도달거리, 도달=<{SR}. 정책 무관. (각 = world −x 기준 ±회전)")
    print(f"\n{'stage':<10} {'side':<4} {'best거리':>8} {'도달?':>6}")
    results = {}
    for name, deg in ANGLES.items():
        for side, sgn in [("좌", +1), ("우", -1)]:
            theta = PI + sgn * math.radians(deg)
            bd, bp = cem(env, theta, n_steps)
            reached = bd < SR
            results[(name, side)] = (bd, bp, theta)
            print(f"{name:<10} {side:<4} {bd:>8.3f} {'Y' if reached else 'n':>6}")
    env.close()

    print("\n  판정:")
    for name in ANGLES:
        ds = [results[(name, s)][0] for s in ("좌", "우")]
        ok = all(d < SR for d in ds)
        print(f"   {name}: 좌 {ds[0]:.3f} / 우 {ds[1]:.3f}  → "
              f"{'최적제어 도달 가능' if ok else '최적제어로도 도달 실패 (물리 한계)'}")


if __name__ == "__main__":
    main()
