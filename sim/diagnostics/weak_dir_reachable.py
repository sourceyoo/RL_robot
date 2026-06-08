"""약방향 target 도달이 물리적으로 가능한 ctrl 이 존재하는가 (정책 무관, read-only).

mirror_physics_test 결론: 강방향 성공 동작의 거울상은 약방향 0% 도달 (물리 좌우 비대칭 A).
미해결: "거울상이 안 됨" ≠ "약방향 도달 절대 불가". 약방향 전용 다른 동작은 가능할 수도.

방법: 강/약 target 을 고정하고 freq×amp×offset constant action 그리드를 open-loop 주입,
도달하는 조합이 하나라도 있는지. 강방향을 동일 그리드로 같이 돌려 baseline.
판정:
 약방향 도달 조합 > 0 → 물리적으로 가능, 정책이 미발견 (처방: 탐색·약방향 데이터↑·mirror aug).
 약방향 0 & 강방향 > 0 → constant 로는 약방향 불가(강방향만 됨). 시변 필요 or 물리 차단 강함.
 양쪽 0 → constant 자체 부족 (parallel 은 시변 필수) — 별도 시변 탐색 필요.
사용: python3 diagnostics/weak_dir_reachable.py
"""
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv, IDX_YAW, FREQ_MIN, FREQ_MAX, AMP_MIN, AMP_MAX

PI = np.pi
SEED = 4000
OFF = PI * 12.0 / 180.0          # s3a 대표 offset 12°
FREQS = [4.0, 5.0, 6.0]
AMPS = [0.5, 0.75, 1.0]
OFFSETS = list(np.round(np.linspace(-1.0, 1.0, 9), 3))


def set_target(env, theta):
    r = float(np.linalg.norm(env._target_pos()[:2]))   # reset 시 target_radius
    gid = env._target_geom_id
    env.model.geom_pos[gid] = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
    mujoco.mj_forward(env.model, env.data)
    # _yaw0/_target_offset/_turn_sign 재계산 (reset 복제)
    yaw0 = float(env.data.qpos[IDX_YAW])
    hd0 = np.array([-np.cos(yaw0), np.sin(yaw0)])
    rel0 = env._target_pos()[:2] - env._torso_pos()[:2]
    n = float(np.linalg.norm(rel0))
    env._yaw0 = yaw0
    env._turn_sign = -float(np.sign(hd0[0] * rel0[1] - hd0[1] * rel0[0])) if n > 1e-6 else 0.0
    env._target_offset = float(np.arccos(np.clip(float(np.dot(hd0, rel0 / n)), -1.0, 1.0))) if n > 1e-6 else 0.0
    env._prev_distance = env._distance_to_target()
    env._prev_pos = env._torso_pos()[:2].copy()


def run_const(env, theta, freq, amp, offset):
    env.reset(seed=SEED)
    set_target(env, theta)
    ts = int(np.sign(env._turn_sign)) if env._turn_sign != 0 else 1
    c0 = 2.0 * (freq - FREQ_MIN) / (FREQ_MAX - FREQ_MIN) - 1.0
    c1 = 2.0 * (amp - AMP_MIN) / (AMP_MAX - AMP_MIN) - 1.0
    action = np.array([c0, c1, offset], dtype=np.float32)
    best = env._distance_to_target()
    reached = False
    term = trunc = False
    while not (term or trunc):
        _, _, term, trunc, info = env.step(action)
        best = min(best, env._distance_to_target())
        if bool(info.get("reached", False)):
            reached = True
            break
    return ts, reached, best


def sweep(env, theta, label):
    ts = None
    hits, best_all = [], 1e9
    for f in FREQS:
        for a in AMPS:
            for o in OFFSETS:
                ts, reached, best = run_const(env, theta, f, a, o)
                best_all = min(best_all, best)
                if reached:
                    hits.append((f, a, o, best))
    print(f"\n  [{label}] θ={np.degrees(theta):.1f}°, turn_sign={ts:+d}  ({len(FREQS)*len(AMPS)*len(OFFSETS)} 조합)")
    print(f"    도달 조합 {len(hits)}개,  전체 최소 거리 {best_all:.3f} (성공<0.08)")
    for f, a, o, b in hits[:8]:
        print(f"      freq={f} amp={a} offset={o:+.2f}  최종근접 {b:.3f}")
    return len(hits)


def main():
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                      target_theta_offset_range=(OFF, OFF))
    print(f"[weak dir reachable] constant ctrl grid open-loop. offset={np.degrees(OFF):.1f}°")
    # θ=π+off 와 θ=π-off 중 어느 게 강(turn_sign+1)/약(-1) 인지는 sweep 가 turn_sign 출력
    n_a = sweep(env, PI + OFF, "A: θ=π+off")
    n_b = sweep(env, PI - OFF, "B: θ=π-off")
    env.close()
    strong, weak = (n_a, n_b)
    print("\n  판정:")
    if min(n_a, n_b) > 0:
        print("   → 양 방향 모두 도달 조합 존재. 약방향도 물리적 가능 → 정책 미발견(B 처방: 탐색·약방향 데이터↑).")
    elif max(n_a, n_b) > 0:
        print("   → 한 방향만 도달 조합 존재(강방향). 약방향은 constant 로 불가 → 시변 필요 or 물리 차단 강함.")
    else:
        print("   → 양쪽 0. constant 자체 부족(parallel 은 시변 필수). 시변 패턴 탐색 별도 필요.")


if __name__ == "__main__":
    main()
