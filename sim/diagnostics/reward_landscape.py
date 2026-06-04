"""보상 landscape 진단 (read-only, 학습 trigger 아님).

질문: v15(복합 par = head_forward·align·turn_ratio + lat_pen 항상 켜기)가 게걸음/과소회전을
정말 비매력화하는가? — v14 rollout 실측 state에 두 reward 식을 적용해 학습 전 정량 확인.

각 step에서 env 내부 state(head_forward·align·turn_ratio·v_lat_avg·progress)를 측정하고
 현행(v14):  par = PAR_W·max(hf,0)·max(align,0)        / lat_pen = LAT_W·|vlat_avg|·(1−turn_ratio)·dt
 v15(후보):  par = PAR_W·max(hf,0)·max(align,0)·turn_ratio / lat_pen = LAT_W·|vlat_avg|·dt   (항상 켬)
두 식의 ep 누적을 비교. v15에서 게걸음(turn_ratio≈0) par→0, lat_pen 부활 이면 게걸음 비매력화.
사용: python3 diagnostics/reward_landscape.py [card]
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv

PI = np.pi
DT = 0.084
N_EP = 30
PAR_W, LAT_W, PROGRESS_P = 12.0, 3.0, 10
STAGES = {"s3a_arc15": (PI / 12, PI / 12), "s3b_arc30": (PI / 6, PI / 6)}


def analyze(model_path, off):
    model = SAC.load(str(model_path), device="cpu")
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                      target_theta_offset_range=off)
    acc = {k: [] for k in ("par_v14", "par_v15", "lat_v14", "lat_v15", "progress", "reach_v14", "reach_v15", "tr_mean")}
    for ep in range(N_EP):
        obs, _ = env.reset(seed=5000 + ep)
        prev_pos = env._torso_pos()[:2].copy()
        prev_dist = env._distance_to_target()
        s = {k: 0.0 for k in acc}
        trs = []
        term = trunc = reached = False
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            xy = env._torso_pos()[:2]
            yaw = float(env.data.qpos[2])
            hd = np.array([-np.cos(yaw), np.sin(yaw)])
            tgt = env._target_pos()[:2]
            rel = tgt - xy
            rn = np.linalg.norm(rel)
            align = float(np.dot(hd, rel / rn)) if rn > 1e-6 else 0.0
            tr = float(np.clip((yaw - env._yaw0) * env._turn_sign / env._target_offset, 0.0, 1.0)) \
                if env._target_offset > 1e-3 else 1.0
            delta = xy - prev_pos
            hf = float(np.dot(delta, hd))
            dn = float(np.linalg.norm(delta))
            cos_m = hf / (dn + 1e-8)
            dist = env._distance_to_target()
            dist_red = prev_dist - dist
            progress = dist_red * (np.clip(cos_m, 0, 1) ** PROGRESS_P) if dist_red > 0 else dist_red
            vlat_avg = float(np.mean(env._v_lat_buffer))  # step에서 갱신됨
            # 두 식
            s["par_v14"] += PAR_W * max(hf, 0.0) * max(align, 0.0)
            s["par_v15"] += PAR_W * max(hf, 0.0) * max(align, 0.0) * tr
            s["lat_v14"] += LAT_W * abs(vlat_avg) * (1.0 - tr) * DT
            s["lat_v15"] += LAT_W * abs(vlat_avg) * DT
            s["progress"] += progress * 5.0
            trs.append(tr)
            reached = bool(info.get("reached", False))
            prev_pos = xy.copy(); prev_dist = dist
            if reached:
                # reach_gate: FLOOR + (1-FLOOR)*tr  (v15는 tr 곱 강화 동일식; 여기선 현행 reach만)
                s["reach_v14"] += 10.0 * (0.4 + 0.6 * tr)
                s["reach_v15"] += 10.0 * (0.4 + 0.6 * tr)
                break
        if not reached:
            continue
        s["tr_mean"] = float(np.mean(trs))
        for k in acc:
            acc[k].append(s[k])
    env.close()
    return {k: np.array(v) for k, v in acc.items()}


def report(tag, a):
    n = len(a["par_v14"])
    if n == 0:
        print(f"  {tag}: 도달 ep 없음"); return
    m = {k: a[k].mean() for k in a}
    print(f"  {tag}: 도달 {n}/{N_EP}ep, turn_ratio 평균 {m['tr_mean']:.2f}")
    print(f"    par_reward 누적:   현행(v14) {m['par_v14']:7.2f}  →  복합(v15) {m['par_v15']:7.2f}  "
          f"({'게걸음 비매력화 ✓' if m['par_v15'] < m['par_v14'] * 0.5 else '변화 작음'})")
    print(f"    lat_pen 누적(벌):  현행(v14) {m['lat_v14']:7.2f}  →  항상켬(v15) {m['lat_v15']:7.2f}  "
          f"(sideslip 페널티 {m['lat_v15'] / (m['lat_v14'] + 1e-6):.1f}x 강화)")
    print(f"    참고: progress 누적 {m['progress']:.2f}  reach 누적 {m['reach_v14']:.2f}")
    # v15 순효과: par 손실 + lat 추가 벌
    net = (m['par_v15'] - m['par_v14']) - (m['lat_v15'] - m['lat_v14'])
    print(f"    → v15 적용 시 이 행동의 보상 순변화: {net:+.2f}  (par {m['par_v15'] - m['par_v14']:+.2f} + lat벌 {-(m['lat_v15'] - m['lat_v14']):+.2f})")


print(f"[reward landscape] N_EP={N_EP}. v14 rollout 에 v14 vs v15 reward 식 적용 비교.")
CARD = sys.argv[1] if len(sys.argv) > 1 else "m4_cpg_v14"
print(f"[card={CARD}]")
base = Path(__file__).parent.parent / "runs" / CARD / "seed0"
for stage, off in STAGES.items():
    mp = base / stage / "model_best.zip"
    if not mp.exists():
        mp = base / stage / "model.zip"
    report(f"{stage}", analyze(mp, off)) if mp.exists() else print(f"  {stage}: 모델 없음")
