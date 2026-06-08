"""좌우 물리 대칭성 테스트 (read-only, 학습 trigger 아님).

질문: 약방향(좌) 선회 도달 0% 가 (A) 물리 한계(fin 비대칭) 인가 (B) 정책이 한쪽만 학습 인가?

방법: 강방향(turn_sign=+1) 에서 정책이 성공한 ep 의 action[freq,amp,offset] 시퀀스를 기록 후
 (a) 그대로 open-loop 재생 → 강방향 target. open-loop 발산 안 하는지 sanity (closed≈open).
 (b) offset 부호 반전 + target y-mirror 로 open-loop 재생 → 약방향 target.
판정:
 (a) 도달 O & (b) 도달 O → 물리 좌우 대칭, 약방향 부진은 학습 편향 (B). 처방: 탐색·mirror aug·장기학습.
 (a) 도달 O & (b) 도달 X → 물리 좌우 비대칭 (A, fin). 보상으론 한계.
 (a) 도달 X → open-loop 발산, 이 테스트 무효 → closed-loop mirror 필요.

좌우 미러 근사: 회전 bias = offset(평균 tail 각) 부호 반전. sin 진동은 좌우대칭(정상상태 추진 위상무관).
사용: python3 diagnostics/mirror_physics_test.py [card] [seedN]
"""
import sys
from pathlib import Path

import numpy as np
import mujoco
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv, IDX_YAW, FREQ_MIN, FREQ_MAX, AMP_MIN, AMP_MAX

PI = np.pi
N_PICK = 25          # 강방향 성공 ep 후보 탐색 수
STAGE = ("s3a_arc15", (PI * 9.2 / 180.0, PI / 12))


def recompute_turn_state(env):
    """target geom_pos 변경 후 reset 의 _yaw0/_target_offset/_turn_sign/_prev 재계산 (reset 코드 복제)."""
    yaw0 = float(env.data.qpos[IDX_YAW])
    head_dir0 = np.array([-np.cos(yaw0), np.sin(yaw0)])
    rel0 = env._target_pos()[:2] - env._torso_pos()[:2]
    rel0_norm = float(np.linalg.norm(rel0))
    env._yaw0 = yaw0
    if rel0_norm > 1e-6:
        align0 = float(np.dot(head_dir0, rel0 / rel0_norm))
        env._target_offset = float(np.arccos(np.clip(align0, -1.0, 1.0)))
        env._turn_sign = -float(np.sign(head_dir0[0] * rel0[1] - head_dir0[1] * rel0[0]))
    else:
        env._target_offset = 0.0
        env._turn_sign = 0.0
    env._prev_distance = env._distance_to_target()
    env._prev_pos = env._torso_pos()[:2].copy()


def rollout_record(env, model, seed):
    """closed-loop 정책 rollout. action 시퀀스·도달 반환."""
    obs, _ = env.reset(seed=seed)
    ts = int(np.sign(env._turn_sign)) if env._turn_sign != 0 else 1
    acts, reached = [], False
    term = trunc = False
    while not (term or trunc):
        a, _ = model.predict(obs, deterministic=True)
        acts.append(np.array(a, dtype=np.float32))
        obs, _, term, trunc, info = env.step(a)
        reached = bool(info.get("reached", False))
        if reached:
            break
    return ts, acts, reached


def replay(env, seed, acts, mirror):
    """동일 seed reset 후 기록 action open-loop 재생. mirror=True 면 target y반전 + offset 부호반전."""
    env.reset(seed=seed)
    if mirror:
        gid = env._target_geom_id
        tp = env.model.geom_pos[gid].copy()
        tp[1] = -tp[1]                     # target y-mirror (sign 반전)
        env.model.geom_pos[gid] = tp
        mujoco.mj_forward(env.model, env.data)
        recompute_turn_state(env)
    reached = False
    term = trunc = False
    i = 0
    while not (term or trunc):
        a = acts[i] if i < len(acts) else acts[-1]   # 기록보다 길면 마지막 유지
        if mirror:
            a = a.copy(); a[2] = -a[2]                # offset 부호 반전
        _, _, term, trunc, info = env.step(a)
        reached = bool(info.get("reached", False))
        i += 1
        if reached:
            break
    return reached


def replay_full_mirror(env, seed, acts):
    """완전 미러: target y반전 + 매 mj_step 의 CPG ctrl 전체 부호 반전(-ctrl). 직접 mj_step (env.step 우회).
    offset 뿐 아니라 sin 진동 위상까지 거울상 → 좌우 물리 대칭이면 약방향 정확 미러 궤적."""
    env.reset(seed=seed)
    gid = env._target_geom_id
    tp = env.model.geom_pos[gid].copy(); tp[1] = -tp[1]
    env.model.geom_pos[gid] = tp
    mujoco.mj_forward(env.model, env.data)
    recompute_turn_state(env)
    timestep = env.model.opt.timestep
    phase = 0.0
    reached = False
    for k in range(len(acts)):
        c = np.clip(acts[k], -1.0, 1.0)
        freq = FREQ_MIN + (float(c[0]) + 1.0) * 0.5 * (FREQ_MAX - FREQ_MIN)
        amp = AMP_MIN + (float(c[1]) + 1.0) * 0.5 * (AMP_MAX - AMP_MIN)
        offset = float(c[2])
        for _ in range(env.frame_skip):
            ctrl = amp * np.sin(2.0 * np.pi * phase) + offset
            env.data.ctrl[0] = float(np.clip(-ctrl, -1.0, 1.0))   # 전체 부호 반전
            mujoco.mj_step(env.model, env.data)
            phase = (phase + freq * timestep) % 1.0
        if env._distance_to_target() < env.success_radius:
            reached = True
            break
    return reached


def main():
    card = sys.argv[1] if len(sys.argv) > 1 else "m4_cpg_v16"
    seedn = sys.argv[2] if len(sys.argv) > 2 else "seed0"
    base = Path(__file__).parent.parent / "runs" / card / seedn
    stage, off = STAGE
    mp = base / stage / "model_best.zip"
    if not mp.exists():
        mp = base / stage / "model.zip"
    print(f"[mirror physics test] card={card} {seedn} {stage}")
    print(f"  model={mp.name}, 강방향(turn_sign=+1) 성공 ep 를 좌우 미러 open-loop 재생.")
    model = SAC.load(str(mp), device="cpu")
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                      target_theta_offset_range=off)

    n_strong = 0
    rep_strong_ok = 0   # (a) 그대로 재생 도달
    rep_mirror_ok = 0   # (b) offset만 반전 미러 도달
    rep_full_ok = 0     # (b2) 완전 ctrl 반전 미러 도달
    offs = []           # 강방향 성공 action 의 offset 절댓값 (미러 의미 판단용)
    for ep in range(N_PICK):
        seed = 4000 + ep
        ts, acts, reached = rollout_record(env, model, seed)
        if ts != +1 or not reached:
            continue                      # 강방향 성공 ep 만 사용
        n_strong += 1
        offs.append(float(np.mean([abs(np.clip(a[2], -1, 1)) for a in acts])))
        rep_strong_ok += int(replay(env, seed, acts, mirror=False))
        rep_mirror_ok += int(replay(env, seed, acts, mirror=True))
        rep_full_ok += int(replay_full_mirror(env, seed, acts))
    env.close()

    if n_strong == 0:
        print("  강방향 성공 ep 없음 — seed 범위 조정 필요."); return
    print(f"\n  강방향 성공 ep: {n_strong}개,  성공 action 의 |offset| 평균 {np.mean(offs):.3f} (회전 bias 크기)")
    print(f"  (a) 그대로 open-loop 재생     → 강방향 target 도달 {100*rep_strong_ok/n_strong:4.0f}%  (sanity: 높아야 open-loop 타당)")
    print(f"  (b) offset만 반전 + target미러 → 약방향 target 도달 {100*rep_mirror_ok/n_strong:4.0f}%")
    print(f"  (b2) 완전 ctrl 반전 + target미러 → 약방향 target 도달 {100*rep_full_ok/n_strong:4.0f}%  (핵심, 완전 거울상)")
    print("\n  판정:")
    sa = rep_strong_ok / n_strong; mb = max(rep_mirror_ok, rep_full_ok) / n_strong
    if sa < 0.5:
        print("   ⚠ (a) 낮음 → open-loop 발산. 이 테스트 무효. closed-loop mirror 필요.")
    elif mb >= 0.5:
        print("   → (a)O (b/b2)O: 물리 좌우 대칭. 약방향 부진 = 학습 편향(B). 처방: 탐색·mirror aug·장기학습.")
    else:
        print("   → (a)O (b/b2)X: 물리 좌우 비대칭(A, fin). 완전 거울상도 약방향 도달 못 함 → 보상으론 한계.")


if __name__ == "__main__":
    main()
