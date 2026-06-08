"""좌/우 회전 이득 곡선: amp·offset 키우면 좌 회전이 실제로 느는가 (open-loop, read-only).

가설: 정책이 좌에서 작은 amp 쓰는 건 '큰 amp·offset 줘도 좌 회전이 안 늘어서' 합리적 선택이다.
검증: 좌/우 mirror target 에 amp×offset grid 고정 60step 주입 → turn_ratio·yaw_net·슬립비.
 큰 amp/offset 에서 좌 turn_ratio 가 우만큼 늘면 → 정책이 큰 거 써야 함(비합리, 처방 여지).
 좌가 포화/저조(우보다 한참 낮음) → 큰 amp 무익 = 좌선회 물리 한계 (정책 합리, 처방 제약).
사용: python3 diagnostics/left_turn_gain.py
"""
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv, IDX_X, IDX_Y, IDX_YAW, FREQ_MIN, FREQ_MAX, AMP_MIN, AMP_MAX

PI = np.pi
SEED = 4000
FREQ = 5.0
STEPS = 60
AMPS = [0.3, 0.5, 0.75, 1.0]
OFFS = [0.2, 0.4, 0.6, 0.8]
TARGET_OFF = {"s3a(12°)": PI * 12 / 180, "s3b(30°)": PI * 30 / 180}


def amp_c(a):
    return 2.0 * (a - AMP_MIN) / (AMP_MAX - AMP_MIN) - 1.0


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


def run(env, theta, amp, off_signed):
    """고정 STEPS 주입 (도달 무관). turn_ratio·yaw_net·슬립비."""
    env.reset(seed=SEED)
    set_target(env, theta)
    yaw0, toff, tsign = env._yaw0, env._target_offset, env._turn_sign
    a = np.array([freq_c(FREQ), amp_c(amp), off_signed], dtype=np.float32)
    vf, vl, tr = [], [], []
    for _ in range(STEPS):
        _, _, term, trunc, info = env.step(a)
        yaw = float(env.data.qpos[IDX_YAW])
        hd = np.array([-np.cos(yaw), np.sin(yaw)])
        perp = np.array([-hd[1], hd[0]])
        v = np.array([float(env.data.qvel[IDX_X]), float(env.data.qvel[IDX_Y])])
        vf.append(float(np.dot(v, hd))); vl.append(abs(float(np.dot(v, perp))))
        tr.append(float(np.clip((yaw - yaw0) * tsign / toff, 0, 1)) if toff > 1e-3 else 1.0)
        if term or trunc:
            break
    vfm = float(np.mean(vf))
    yaw_net = float(np.degrees((float(env.data.qpos[IDX_YAW]) - yaw0) * tsign))
    return float(np.mean(tr)), yaw_net, float(np.mean(vl)) / (abs(vfm) + 1e-9)


def main():
    print(f"[좌/우 회전 이득] amp×offset 고정 {STEPS}step, freq={FREQ}Hz. 도달 무관, 회전 효율 곡선.")
    print("  질문: 큰 amp·offset 이 좌 회전(turn_ratio·yaw_net)을 우만큼 늘리는가?")
    for name, off in TARGET_OFF.items():
        env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                          target_theta_offset_range=(off, off))
        print(f"\n  ### {name}  (yaw_net=target쪽 회전 deg, 음수=반대로)")
        print(f"    {'offset':>7} | " + " | ".join(f"amp{a:<4}" for a in AMPS))
        for metric, lab in [("yaw", "yaw_net(deg)"), ("tr", "turn_ratio")]:
            print(f"    [{lab}]")
            for o in OFFS:
                rowL, rowR = [], []
                for amp in AMPS:
                    trL, yL, _ = run(env, PI + off, amp, +o)   # 좌
                    trR, yR, _ = run(env, PI - off, amp, -o)   # 우
                    if metric == "yaw":
                        rowL.append(f"{yL:+5.1f}"); rowR.append(f"{yR:+5.1f}")
                    else:
                        rowL.append(f"{trL:.2f}"); rowR.append(f"{trR:.2f}")
                print(f"    좌 o+{o:<4} | " + " | ".join(f"{x:>6}" for x in rowL))
                print(f"    우 o-{o:<4} | " + " | ".join(f"{x:>6}" for x in rowR))
        env.close()
    print("\n  해석: 좌 yaw_net/turn_ratio 가 amp·offset 키워도 우보다 한참 낮고 포화면")
    print("  → 큰 amp 무익 = 좌선회 물리 한계 (정책의 작은 amp 가 합리적). 우만큼 늘면 → 정책 비합리(처방 여지).")


if __name__ == "__main__":
    main()
