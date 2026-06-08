"""좌회전이 시변(time-varying) ctrl 로는 도달 가능한가 (정책 무관, read-only).

weak_dir_reachable(constant)은 좌회전 도달이 marginal(0.079, 경계 스침)이었음. constant 는
"틀기(+offset, 추진죽음)"와 "추진(offset≈0)"을 동시에 못 함. 시변 2-phase 로 분리하면?
 phase1 (t<T1): offset=+o1, 작은 amp → 머리를 좌로 틀기 (추진 희생)
 phase2 (t≥T1): offset=o2(≈0), amp=1.0 → 머리 방향 유지하며 직진 추진
좌(turn_sign=-1, θ=π+off) / 우(turn_sign=+1, θ=π-off) 둘 다 grid open-loop.
판정:
 좌회전 시변 도달 다수 + 여유(거리<0.06) → 문 있음 → 보상으로 유도 가능.
 좌회전 시변도 0/marginal → 문 거의 없음 → 물리 한계, 보상으론 한계.
사용: python3 diagnostics/timevarying_reachable.py [card] [seedN]  (모델 불필요 — 정책 무관)
"""
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv, IDX_YAW, FREQ_MIN, FREQ_MAX, AMP_MIN, AMP_MAX

PI = np.pi
SEED = 4000
OFF = PI * 12.0 / 180.0
FREQ = 5.0
A_TURN, A_DRIVE = 0.3, 1.0          # 틀기 작은 amp / 직진 큰 amp
O1S = [0.5, 0.7, 0.9]              # 틀기 bias
T1S = [50, 100, 200, 350]         # 틀기 지속 step
O2S = [-0.3, -0.1, 0.1]           # 직진 bias (머리방향 미세유지)


def amp_c(amp):
    return 2.0 * (amp - AMP_MIN) / (AMP_MAX - AMP_MIN) - 1.0


def freq_c(f):
    return 2.0 * (f - FREQ_MIN) / (FREQ_MAX - FREQ_MIN) - 1.0


def set_target(env, theta):
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


def run_2phase(env, theta, o1, T1, o2):
    env.reset(seed=SEED)
    set_target(env, theta)
    ts = int(np.sign(env._turn_sign)) if env._turn_sign != 0 else 1
    fc = freq_c(FREQ)
    best = env._distance_to_target()
    reached = False
    t = 0
    term = trunc = False
    while not (term or trunc):
        if t < T1:
            action = np.array([fc, amp_c(A_TURN), o1], dtype=np.float32)
        else:
            action = np.array([fc, amp_c(A_DRIVE), o2], dtype=np.float32)
        _, _, term, trunc, info = env.step(action)
        best = min(best, env._distance_to_target())
        if bool(info.get("reached", False)):
            reached = True
            break
        t += 1
    return ts, reached, best


def sweep(env, theta, label):
    ts = None
    hits, best_all = [], 1e9
    for o1 in O1S:
        for T1 in T1S:
            for o2 in O2S:
                ts, reached, best = run_2phase(env, theta, o1, T1, o2)
                best_all = min(best_all, best)
                if reached:
                    hits.append((o1, T1, o2, best))
    n = len(O1S) * len(T1S) * len(O2S)
    print(f"\n  [{label}] θ={np.degrees(theta):.1f}°, turn_sign={ts:+d}  ({n} 조합, 2-phase 틀기→직진)")
    print(f"    도달 조합 {len(hits)}개,  전체 최소거리 {best_all:.3f} (성공<0.08, 여유=<0.06)")
    for o1, T1, o2, b in sorted(hits, key=lambda x: x[3])[:8]:
        print(f"      o1(틀기)={o1} T1={T1}step o2(직진)={o2:+.1f}  최종근접 {b:.3f}")
    return len(hits), best_all


def main():
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                      target_theta_offset_range=(OFF, OFF))
    print(f"[timevarying reachable] 2-phase open-loop. offset={np.degrees(OFF):.1f}° (정책 무관)")
    nL, bL = sweep(env, PI + OFF, "좌회전 θ=π+off")
    nR, bR = sweep(env, PI - OFF, "우회전 θ=π-off")
    env.close()
    print("\n  판정:")
    if nL > 0 and bL < 0.06:
        print(f"   → 좌회전 시변 여유 도달({nL}개, 최소 {bL:.3f}). 문 있음 → 보상으로 유도 가능.")
    elif nL > 0:
        print(f"   → 좌회전 시변 도달하나 marginal(최소 {bL:.3f}). 문이 좁음 → 보상 유도 어려움.")
    else:
        print(f"   → 좌회전 시변 도달 0 (최소 {bL:.3f}). 문 거의 없음 → 물리 한계, 보상으론 한계.")
    print(f"   (우회전 baseline: 도달 {nR}개, 최소 {bR:.3f})")


if __name__ == "__main__":
    main()
