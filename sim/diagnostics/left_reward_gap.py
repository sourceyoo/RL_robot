"""D-H2: 좌 target 에서 parallel vs sideslip 누적보상 비교 (보상 빈틈 H2 진단, read-only).

D-H1 으로 좌 parallel 물리 가능 확정(H1 기각). 그런데 정책은 게걸음(sideslip)을 한다.
질문: 좌 target 에서 sideslip 도달의 누적보상이 parallel 도달 ≥ 인가? 그렇다면 정책이 더 쉬운
게걸음을 택하는 게 합리적 = H2(보상 빈틈). 어느 항이 sideslip 을 떠받치는지 = 처방 타겟.

방법:
 P (parallel) : D-H1 확인된 좌 parallel constant action (큰 amp + 큰 +offset).
 S (sideslip) : 정책이 실제 쓰는 게걸음 재현 (작은 amp + |offset|>amp).
 둘 다 open-loop 로 좌 target 주입, reward 항 누적 분해 (self-check mismatch≈0 검증).
 + closed-loop 정책 좌 ep 항분해 (상수 S 가 정책 거동 대표하는지 교차검증).
판정: 누적(S) ≥ 누적(P) (차<5%) → H2 확정. 누적(P) > 누적(S) (>10%) → H2 기각(H3 탐색).
사용: python3 diagnostics/left_reward_gap.py
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
# P: D-H1 에서 s3a 좌 parallel 도달 (turn_ratio 0.58·슬립비 0.24). S: 정책 s3a 좌 sideslip (action_usage).
P_ACTION = (5.0, 0.75, +0.75)
S_ACTION = (5.82, 0.218, +0.678)
TERM_KEYS = ['progress', 'reach', 'par', 'align', 'yaw_sign',
             'time_pen', 'back_pen', 'smooth_pen', 'lat_pen', 'stop_pen']


def amp_c(a):
    return 2.0 * (a - AMP_MIN) / (AMP_MAX - AMP_MIN) - 1.0


def freq_c(f):
    return 2.0 * (f - FREQ_MIN) / (FREQ_MAX - FREQ_MIN) - 1.0


def set_left_target(env, off):
    """좌 target (turn_sign=-1): θ=π+off."""
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


def reward_terms(env, clipped, prev_pos, prev_dist, prev_action, step_count):
    """fish_env.step reward 공식 재현 + 항 분해 (barrier_quant 복제, self-check 검증됨)."""
    torso_xy = env._torso_pos()[:2]
    distance = env._distance_to_target()
    reached = distance < env.success_radius
    yaw = float(env.data.qpos[IDX_YAW])
    head_dir = np.array([-np.cos(yaw), np.sin(yaw)])
    rel = env._target_pos()[:2] - torso_xy
    rel_norm = float(np.linalg.norm(rel))
    align = float(np.dot(head_dir, rel / rel_norm)) if rel_norm > 1e-6 else 0.0
    yaw_err_sign = float(np.sign(head_dir[0] * rel[1] - head_dir[1] * rel[0])) if rel_norm > 1e-6 else 0.0
    PROGRESS_P = 10
    delta = torso_xy - prev_pos
    head_forward = float(np.dot(delta, head_dir))
    cos_motion = head_forward / (float(np.linalg.norm(delta)) + 1e-8)
    dist_reduction = prev_dist - distance
    motion_gate = float(np.clip(cos_motion, 0, 1)) ** PROGRESS_P
    progress = dist_reduction * motion_gate if dist_reduction > 0 else dist_reduction
    YAW_SIGN_W = 0.020; TIME_PEN_W = 4e-5; BACK_PEN_W = 40.0; SMOOTH_W = 0.05
    STOP_PEN_W = 4.0; V_STOP = 0.05; STOP_GRACE = 3
    yaw_rate = float(env.data.qvel[IDX_YAW])
    current_time = step_count * env.dt
    v_world = np.array([float(env.data.qvel[IDX_X]), float(env.data.qvel[IDX_Y])])
    v_norm = float(np.linalg.norm(v_world))
    back_pen = 0.0
    if v_norm > 0.05:
        cos_hv = float(np.dot(head_dir, v_world / v_norm))
        if cos_hv < 0.0:
            back_pen = BACK_PEN_W * (-cos_hv) * v_norm
    stop_pen = 0.0
    if (not reached) and (step_count >= STOP_GRACE) and (v_norm < V_STOP):
        stop_pen = STOP_PEN_W * (V_STOP - v_norm)
    smooth_pen = SMOOTH_W * float(np.sum((clipped - prev_action) ** 2))
    if env._target_offset > 1e-3:
        turn_ratio = float(np.clip((yaw - env._yaw0) * env._turn_sign / env._target_offset, 0, 1))
    else:
        turn_ratio = 1.0
    reach_gate = 0.4 + 0.6 * turn_ratio
    par = 12.0 * max(head_forward, 0.0) * max(align, 0.0) * turn_ratio
    lat_pen = 4.0 * abs(float(np.mean(env._v_lat_buffer)))
    dt = env.dt
    terms = dict(
        progress=progress * 5.0,
        reach=(10.0 * reach_gate if reached else 0.0),
        par=par,
        align=(env.align_weight * align * turn_ratio * (1.0 if v_norm > 0.05 else 0.0)) * dt,
        yaw_sign=(YAW_SIGN_W * yaw_rate * yaw_err_sign) * dt,
        time_pen=-(TIME_PEN_W * distance * current_time) * dt,
        back_pen=-back_pen * dt,
        smooth_pen=-smooth_pen * dt,
        lat_pen=-lat_pen * dt,
        stop_pen=-stop_pen * dt,
    )
    return terms, sum(terms.values()), reached


def run_const(off, freq, amp, offset, label):
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                      target_theta_offset_range=(off, off))
    env.reset(seed=SEED)
    set_left_target(env, off)
    a = np.array([freq_c(freq), amp_c(amp), offset], dtype=np.float32)
    acc = {k: 0.0 for k in TERM_KEYS}
    best = env._distance_to_target(); reached = False; mm = 0.0
    term = trunc = False
    while not (term or trunc):
        clipped = np.clip(a, -1, 1)
        pp = env._prev_pos.copy(); pd = env._prev_distance
        pa = env._prev_action.copy(); sc = env._step_count
        _, rew_env, term, trunc, info = env.step(a)
        terms, total, rchd = reward_terms(env, clipped, pp, pd, pa, sc)
        mm = max(mm, abs(total - rew_env))
        for k in acc:
            acc[k] += terms[k]
        best = min(best, env._distance_to_target())
        if rchd:
            reached = True; break
    env.close()
    return dict(label=label, acc=acc, total=sum(acc.values()), reached=reached, best=best, mm=mm)


def run_policy(stage, off, n_left=20):
    """closed-loop 정책 좌 ep 항분해 (상수 S 교차검증). 좌 ep n_left 개 평균."""
    base = Path(__file__).parent.parent / "runs" / "m4_cpg_v19" / "seed0" / stage
    mp = base / "model.zip"
    if not mp.exists():
        return None
    model = SAC.load(str(mp), device="cpu")
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                      target_theta_offset_range=(off, off))
    accs, reached_n, got = [], 0, 0
    ep = 0
    mm = 0.0
    while got < n_left and ep < 300:
        obs, _ = env.reset(seed=SEED + ep); ep += 1
        if env._turn_sign >= 0:  # 우 ep skip
            continue
        got += 1
        acc = {k: 0.0 for k in TERM_KEYS}; rchd = False
        term = trunc = False
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            clipped = np.clip(a, -1, 1)
            pp = env._prev_pos.copy(); pd = env._prev_distance
            pa = env._prev_action.copy(); sc = env._step_count
            obs, rew_env, term, trunc, info = env.step(a)
            terms, total, r = reward_terms(env, clipped, pp, pd, pa, sc)
            mm = max(mm, abs(total - rew_env))
            for k in acc:
                acc[k] += terms[k]
            rchd = bool(info.get("reached", False))
            if rchd:
                break
        accs.append(acc); reached_n += int(rchd)
    env.close()
    mean_acc = {k: float(np.mean([a[k] for a in accs])) for k in TERM_KEYS}
    return dict(acc=mean_acc, total=sum(mean_acc.values()),
                reach_rate=100.0 * reached_n / max(got, 1), n=got, mm=mm)


def print_terms(label, total, acc, extra=""):
    print(f"    {label:<22} 누적 {total:+8.3f}  {extra}")
    print("      " + "  ".join(f"{k}={acc[k]:+.3f}" for k in TERM_KEYS))


def main():
    print("[D-H2 좌 parallel vs sideslip 누적보상] open-loop 상수 + closed-loop 정책 교차검증")
    print(f"  P(parallel)={P_ACTION}  S(sideslip)={S_ACTION}")
    for stage, off in STAGES.items():
        print(f"\n{'='*78}\n### {stage}  (θ=π+{np.degrees(off):.0f}°)")
        P = run_const(off, *P_ACTION, "P parallel (const)")
        S = run_const(off, *S_ACTION, "S sideslip (const)")
        print(f"  [open-loop 상수, self-check mm: P {P['mm']:.1e} / S {S['mm']:.1e}]")
        print_terms(P["label"], P["total"], P["acc"], f"도달={'Y' if P['reached'] else 'n'} 근접 {P['best']:.3f}")
        print_terms(S["label"], S["total"], S["acc"], f"도달={'Y' if S['reached'] else 'n'} 근접 {S['best']:.3f}")
        diff = S["total"] - P["total"]
        rel = diff / (abs(P["total"]) + 1e-9)
        print(f"  → 누적(S)-누적(P) = {diff:+.3f}  ({rel*100:+.0f}% of |P|)")
        pol = run_policy(stage, off)
        if pol:
            print(f"  [closed-loop 정책 좌 ep {pol['n']}개 평균, 도달 {pol['reach_rate']:.0f}%, self-check mm {pol['mm']:.1e}]")
            print_terms("정책 좌 ep (교차검증)", pol["total"], pol["acc"])
        # 판정
        if diff >= -0.05 * abs(P["total"]):
            verdict = "H2 확정 (sideslip ≥ parallel → 게걸음이 보상상 합리적)"
        elif diff < -0.10 * abs(P["total"]):
            verdict = "H2 기각 (parallel 우세 → H3 탐색 진단)"
        else:
            verdict = "경계 (차 5~10%)"
        print(f"  ※ 판정({stage}): {verdict}")
    print("\n  주범 항 식별: S에서 reach≈4.0(floor) 유지 + lat_pen 작음 + progress 양수면")
    print("  → reach_gate FLOOR=0.4 가 게걸음 떠받침 / lat_pen 약함 / cos_motion 게이팅이 좌 옆이동 못 막음")


if __name__ == "__main__":
    main()
