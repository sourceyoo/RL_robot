"""좌/우 선회별 reward 항 분해 (read-only, 학습 trigger 아님).

가설: v13~v16 par_reward(PAR_W=12 복합)·lat_pen 강화로 yaw_err_sign 항(·dt 곱해 작음)의
방향-균형 효과가 희석 → 약방향(좌) 선회 압력 부족. 검증: 도달/미도달·좌/우 그룹별로
각 reward 항 step당 평균을 env.step 과 동일 식으로 재계산해 비교.

핵심 확인:
 - par_reward 가 강방향을 떠받치고 약방향엔 작은가 (turn_ratio 낮아서) → 편향 강화 여부
 - yaw_sign 항이 약방향에서 + 압력을 충분히 주는가, par 격차 대비 크기
사용: python3 diagnostics/reward_term_asymmetry.py [card] [seedN]
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv, IDX_X, IDX_Y, IDX_YAW

PI = np.pi
DT = 0.084
N_EP = 100
STAGES = {"s3a_arc15": (PI * 9.2 / 180.0, PI / 12), "s3b_arc30": (PI / 12, PI / 6)}

# fish_env.step 과 동일 상수
PROGRESS_P = 10
YAW_SIGN_W = 0.020
TIME_PEN_W = 4e-5
BACK_PEN_W = 40.0
SMOOTH_W = 0.05
PAR_W = 12.0
TURN_FLOOR = 0.4
LAT_AVG_W = 4.0

TERMS = ["progress", "reach", "par", "align", "yaw_sign", "time_pen", "back_pen", "smooth", "lat_pen", "total"]


def analyze(model_path, off):
    model = SAC.load(str(model_path), device="cpu")
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                      target_theta_offset_range=off)
    aw = float(env.align_weight)
    # 그룹: turn_sign +1 / -1, 각각 도달/미도달 분리
    def newg():
        return dict(n=0, reach=0, eplen=[], sums={t: [] for t in TERMS})
    g = {+1: newg(), -1: newg()}
    for ep in range(N_EP):
        obs, _ = env.reset(seed=4000 + ep)
        ts = int(np.sign(env._turn_sign)) if env._turn_sign != 0 else 1
        acc = {t: 0.0 for t in TERMS}
        term = trunc = reached = False
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            clipped = np.clip(a, -1.0, 1.0).astype(np.float32)
            # step 전 상태 캡처 (env.step 이 갱신하기 전)
            prev_pos = env._torso_pos()[:2].copy()
            prev_dist = env._distance_to_target()
            prev_action = env._prev_action.copy()
            sc = env._step_count
            obs, _, term, trunc, info = env.step(a)
            # step 후 상태로 항 재계산 (fish_env.step 식 그대로)
            xy = env._torso_pos()[:2]
            distance = env._distance_to_target()
            reached = bool(info.get("reached", False))
            yaw = float(env.data.qpos[IDX_YAW])
            hd = np.array([-np.cos(yaw), np.sin(yaw)])
            rel = env._target_pos()[:2] - xy
            rn = float(np.linalg.norm(rel))
            align = float(np.dot(hd, rel / rn)) if rn > 1e-6 else 0.0
            yaw_err_sign = float(np.sign(hd[0] * rel[1] - hd[1] * rel[0])) if rn > 1e-6 else 0.0
            delta = xy - prev_pos
            dn = float(np.linalg.norm(delta))
            hf = float(np.dot(delta, hd))
            cos_m = hf / (dn + 1e-8)
            dist_red = prev_dist - distance
            motion_gate = float(np.clip(cos_m, 0.0, 1.0)) ** PROGRESS_P
            progress = dist_red * motion_gate if dist_red > 0 else dist_red
            yaw_rate = float(env.data.qvel[IDX_YAW])
            cur_t = sc * DT
            v_world = np.array([float(env.data.qvel[IDX_X]), float(env.data.qvel[IDX_Y])])
            v_norm = float(np.linalg.norm(v_world))
            back_pen = 0.0
            if v_norm > 0.05:
                cos_hv = float(np.dot(hd, v_world / v_norm))
                if cos_hv < 0.0:
                    back_pen = BACK_PEN_W * (-cos_hv) * v_norm
            action_diff_sq = float(np.sum((clipped - prev_action) ** 2))
            smooth = SMOOTH_W * action_diff_sq
            if env._target_offset > 1e-3:
                turn_ratio = float(np.clip((yaw - env._yaw0) * env._turn_sign / env._target_offset, 0.0, 1.0))
            else:
                turn_ratio = 1.0
            reach_gate = TURN_FLOOR + (1.0 - TURN_FLOOR) * turn_ratio
            par = PAR_W * max(hf, 0.0) * max(align, 0.0) * turn_ratio
            v_lat_avg = float(np.mean(env._v_lat_buffer))
            lat_pen = LAT_AVG_W * abs(v_lat_avg)
            # 항 누적 (reward 부호 그대로: 페널티는 음수)
            acc["progress"] += progress * 5.0
            acc["reach"] += (10.0 * reach_gate) if reached else 0.0
            acc["par"] += par
            acc["align"] += (aw * align * turn_ratio * (1.0 if v_norm > 0.02 else 0.0)) * DT
            acc["yaw_sign"] += (YAW_SIGN_W * yaw_rate * yaw_err_sign) * DT
            acc["time_pen"] += -TIME_PEN_W * distance * cur_t * DT
            acc["back_pen"] += -back_pen * DT
            acc["smooth"] += -smooth * DT
            acc["lat_pen"] += -lat_pen * DT
            if reached:
                break
        acc["total"] = sum(acc[t] for t in TERMS if t != "total")
        d = g[ts]
        d["n"] += 1
        d["reach"] += int(reached)
        d["eplen"].append(env._step_count)
        for t in TERMS:
            d["sums"][t].append(acc[t])
    env.close()
    return g


def report(tag, g):
    print(f"\n  {tag}:")
    for ts in (+1, -1):
        d = g[ts]
        rr = 100.0 * d["reach"] / d["n"] if d["n"] else 0.0
        el = np.mean(d["eplen"]) if d["eplen"] else 0.0
        print(f"    sign={ts:+d}: {d['n']}ep  도달 {rr:4.0f}%  평균 ep 길이 {el:6.1f} step")
    el1 = np.mean(g[+1]["eplen"]) if g[+1]["eplen"] else 1.0
    el2 = np.mean(g[-1]["eplen"]) if g[-1]["eplen"] else 1.0
    print(f"    {'-'*70}")
    print(f"    {'항':>10}  {'+1 누적':>10} {'+1 /step':>11}  |  {'-1 누적':>10} {'-1 /step':>11}")
    for t in TERMS:
        a = np.array(g[+1]["sums"][t]); b = np.array(g[-1]["sums"][t])
        am = a.mean() if len(a) else 0.0
        bm = b.mean() if len(b) else 0.0
        aps = am / el1; bps = bm / el2
        # step당으로 약방향이 60% 미만이면 표시 (도달 항 reach 제외 — ep길이 무관)
        mark = "  <<< 약방향 /step 부족" if (t in ("par", "progress", "align") and bps < aps * 0.6) else ""
        print(f"    {t:>10}  {am:>10.3f} {aps:>11.5f}  |  {bm:>10.3f} {bps:>11.5f}{mark}")


print(f"[reward term asymmetry] N_EP={N_EP}, deterministic. 좌/우(turn_sign) 별 reward 항 분해.")
CARD = sys.argv[1] if len(sys.argv) > 1 else "m4_cpg_v16"
SEED = sys.argv[2] if len(sys.argv) > 2 else "seed0"
print(f"[card={CARD}, {SEED}]")
base = Path(__file__).parent.parent / "runs" / CARD / SEED
for stage, off in STAGES.items():
    mp = base / stage / "model_best.zip"
    if not mp.exists():
        mp = base / stage / "model.zip"
    report(stage, analyze(mp, off)) if mp.exists() else print(f"  {stage}: 모델 없음")
