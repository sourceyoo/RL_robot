"""D-H3b: 좌 parallel 강제진입 후 정책 유지/회귀 (능력 vs 탐색 분리, read-only).

D-H3a: s3b 좌에서 정책이 offset 부호를 틀림(평균 -0.187, 우회전 방향). 좌선회(+offset, 비효율)
대신 우회전+게걸음(-offset, 쉬움) 국소최적 수렴.
질문: 정책을 +offset parallel 상태(머리 좌로 틀린 turn_ratio↑)로 강제 진입시킨 뒤 놓아주면?
 +offset 유지 & 도달 → 능력은 있고 탐색만 실패 = H3 확정 (처방: 탐색·curriculum).
 -offset 게걸음 회귀 → 국소최적 강고/OOD = H3+ (탐색만으론 부족, 다른 처방).

방법: phase1 = constant [freq5, amp0.9, +offset] K step (머리 좌로 틀기). phase2 = 정책 핸드오버.
사용: python3 diagnostics/left_parallel_handoff.py
"""
import sys
from pathlib import Path

import numpy as np
import mujoco
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import (FishSwimEnv, IDX_X, IDX_Y, IDX_YAW,
                      FREQ_MIN, FREQ_MAX, AMP_MIN, AMP_MAX)

PI = np.pi
SEED = 4000
STAGES = {"s3a_arc15": PI * 12.0 / 180.0, "s3b_arc30": PI * 30.0 / 180.0}
INJECT = (5.0, 0.75, 0.75)   # phase1 좌 parallel 진입 (D-H1 검증된 좌 parallel 조합)
K_LIST = [60, 120, 200]      # phase1 주입 step (머리 좌로 틀기 지속)


def amp_c(a):
    return 2.0 * (a - AMP_MIN) / (AMP_MAX - AMP_MIN) - 1.0


def freq_c(f):
    return 2.0 * (f - FREQ_MIN) / (FREQ_MAX - FREQ_MIN) - 1.0


def decode_off(a):
    return float(np.clip(a, -1.0, 1.0)[2])


def set_left_target(env, off):
    theta = PI + off
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


def metrics(env):
    yaw = float(env.data.qpos[IDX_YAW])
    hd = np.array([-np.cos(yaw), np.sin(yaw)])
    perp = np.array([-hd[1], hd[0]])
    v = np.array([float(env.data.qvel[IDX_X]), float(env.data.qvel[IDX_Y])])
    rel = env._target_pos()[:2] - env._torso_pos()[:2]
    rn = float(np.linalg.norm(rel))
    tr = float(np.clip((yaw - env._yaw0) * env._turn_sign / env._target_offset, 0, 1)) if env._target_offset > 1e-3 else 1.0
    align = float(np.dot(hd, rel / rn)) if rn > 1e-6 else 0.0
    vf = float(np.dot(v, hd)); vl = abs(float(np.dot(v, perp)))
    return tr, align, vf, vl


def run_handoff(model, off, K):
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                      target_theta_offset_range=(off, off))
    obs, _ = env.reset(seed=SEED)
    set_left_target(env, off)
    inj = np.array([freq_c(INJECT[0]), amp_c(INJECT[1]), INJECT[2]], dtype=np.float32)
    # phase1: 좌 parallel 강제 진입 (도달 직전 최대 turn_ratio 기록)
    tr1 = 0.0; al1 = 0.0
    for _ in range(K):
        obs, _, term, trunc, info = env.step(inj)
        tr_now, al_now, _, _ = metrics(env)
        if tr_now > tr1:
            tr1, al1 = tr_now, al_now
        if term or trunc or bool(info.get("reached", False)):
            break
    # phase2: 정책 핸드오버
    offs, trs, als, vfs, vls = [], [], [], [], []
    reached = False
    term = trunc = False
    while not (term or trunc):
        a, _ = model.predict(obs, deterministic=True)
        offs.append(decode_off(a))
        obs, _, term, trunc, info = env.step(a)
        tr, al, vf, vl = metrics(env)
        trs.append(tr); als.append(al); vfs.append(vf); vls.append(vl)
        if bool(info.get("reached", False)):
            reached = True; break
    env.close()
    off_mean = float(np.mean(offs)) if offs else 0.0
    off_early = float(np.mean(offs[:20])) if offs else 0.0   # 핸드오버 직후 부호
    slip = float(np.mean(vls)) / (abs(float(np.mean(vfs))) + 1e-9) if vfs else 0.0
    return dict(K=K, tr_handoff=tr1, al_handoff=al1, off_early=off_early, off_mean=off_mean,
                tr_mean=float(np.mean(trs)) if trs else 0.0,
                al_mean=float(np.mean(als)) if als else 0.0, slip=slip, reached=reached)


def main():
    print("[D-H3b 좌 parallel 핸드오버] phase1 +offset 강제진입 → phase2 정책")
    print(f"  주입 action={INJECT} (큰 amp + 큰 +offset). 좌회전 정답=+offset 유지+turn_ratio 유지+도달.")
    base = Path(__file__).parent.parent / "runs" / "m4_cpg_v19" / "seed0"
    for stage, off in STAGES.items():
        mp = base / stage / "model.zip"
        if not mp.exists():
            print(f"\n  {stage}: 모델 없음"); continue
        model = SAC.load(str(mp), device="cpu")
        print(f"\n  ### {stage} (θ=π+{np.degrees(off):.0f}°)")
        print(f"    {'K(주입)':>7} {'진입turn_ratio':>13} {'정책offset초기':>14} {'정책offset평균':>14} "
              f"{'turn_ratio':>10} {'슬립비':>7} {'도달':>5}")
        for K in K_LIST:
            r = run_handoff(model, off, K)
            verd = "+유지" if r["off_early"] > 0.05 else ("-회귀" if r["off_early"] < -0.05 else "~0")
            print(f"    {K:>7} {r['tr_handoff']:>13.3f} {r['off_early']:>+13.3f}({verd}) "
                  f"{r['off_mean']:>+13.3f} {r['tr_mean']:>10.3f} {r['slip']:>7.2f} "
                  f"{'Y' if r['reached'] else 'n':>5}")
    print("\n  판정: 핸드오버 후 정책 offset +유지 & turn_ratio 유지 & 도달 → 능력 있음, 탐색만 실패(H3).")
    print("        offset -회귀 & 게걸음 → 국소최적 강고/OOD (탐색만으론 부족).")


if __name__ == "__main__":
    main()
