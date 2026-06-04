"""m4_cpg_v12 "계단식(stop-and-go / 직진→꺾임→직진)" 진단 (read-only, 학습 trigger 아님).

사용자 지적: v12 가 sideslip 을 줄이려고 '계단식'으로 가는데 parallel mode(돌며 추진) 와 다름.
질문: 머리(yaw)가 매끄럽게 호를 그리며 도나(parallel), 아니면 직진 구간 yaw 고정 + 특정 구간만 급회전(계단식)인가?

지표 (대표 ep 시계열):
  - yaw(t): 매끄러운 단조 증가 = parallel / 계단(평탄→급증→평탄) = sequential
  - |dyaw/dt| 의 집중도: 회전이 전 구간 분산(parallel) vs 소수 step 집중(계단식). gini-like = max구간/전체.
  - v_fwd, v_lat(t): parallel 이면 v_fwd 항상>0 + v_lat 작음. 계단식이면 회전 구간서 v_fwd 급감(거의 정지 후 틂).
사용: python3 diagnostics/staircase_probe.py
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv

PI = np.pi
ACTION_HISTORY_N = 20
DT = 0.084
N_EP = 30
S3B_KW = dict(episode_seconds=60.0, success_radius=0.08, action_history_n=ACTION_HISTORY_N,
              target_theta_offset_range=(PI / 6, PI / 6))  # ±30°


def analyze(model_path):
    model = SAC.load(str(model_path), device="cpu")
    env = FishSwimEnv(**S3B_KW)
    head_turn_deg, slip_deg, turn_concentration, vfwd_min_frac = [], [], [], []
    for ep in range(N_EP):
        obs, _ = env.reset(seed=5000 + ep)
        yaw0 = float(env.data.qpos[2])
        prev_xy = env._torso_pos()[:2].copy()
        yaws, vfwd, vlat = [], [], []
        term = trunc = False
        reached = False
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            xy = env._torso_pos()[:2]
            yaw = float(env.data.qpos[2])
            hd = np.array([-np.cos(yaw), np.sin(yaw)])
            perp = np.array([-hd[1], hd[0]])
            d = (xy - prev_xy) / DT
            vfwd.append(float(np.dot(d, hd)))
            vlat.append(float(np.dot(d, perp)))
            yaws.append(yaw)
            prev_xy = xy.copy()
            reached = bool(info.get("reached", False))
            if reached:
                break
        if not reached:
            continue
        yaws = np.array(yaws); vfwd = np.array(vfwd); vlat = np.array(vlat)
        head_turn_deg.append(np.degrees(yaws[-1] - yaw0))
        # sideslip 각: 진행방향과 머리방향 사이 각의 ep 평균 (이동한 step만)
        spd = np.hypot(vfwd, vlat)
        mv = spd > 1e-3
        if mv.any():
            slip = np.degrees(np.abs(np.arctan2(vlat[mv], vfwd[mv])))
            slip_deg.append(float(np.mean(slip)))
        # 회전 집중도: |dyaw| 누적 중 가장 바쁜 20% step 이 차지하는 비율 (1=완전집중=계단, 0.2=균등=parallel)
        dyaw = np.abs(np.diff(yaws))
        if dyaw.sum() > 1e-4:
            k = max(1, int(0.2 * len(dyaw)))
            top = np.sort(dyaw)[-k:].sum()
            turn_concentration.append(float(top / dyaw.sum()))
        # 전진속도 최저구간: 회전 중 멈추나? (전진속도 5퍼센타일 / 중앙값)
        if np.median(vfwd) > 1e-4:
            vfwd_min_frac.append(float(np.percentile(vfwd, 5) / np.median(vfwd)))
    env.close()
    return (np.array(head_turn_deg), np.array(slip_deg),
            np.array(turn_concentration), np.array(vfwd_min_frac))


def report(tag, ht, sl, tc, vf):
    if len(ht) == 0:
        print(f"  {tag}: 도달 ep 없음"); return
    print(f"  {tag}: 도달 {len(ht)}ep")
    print(f"    머리회전(도)   mean={ht.mean():+6.1f}  (목표 ±30)   |절대값| median={np.median(np.abs(ht)):.1f}")
    print(f"    sideslip(도)   mean={sl.mean():6.1f}  median={np.median(sl):.1f}   (0=완전정렬, 90=순수게걸음)")
    print(f"    회전집중도     mean={tc.mean():.2f}   (0.2=전구간균등=parallel, →1=소수step집중=계단식)")
    print(f"    전진속도최저비 mean={vf.mean():+.2f}  (회전중 멈추면 음수/0근처, parallel이면 0.3~1)")


print(f"[staircase] v12 s3b(±30°) — 계단식 vs parallel 판별. N_EP={N_EP}, 도달 ep만.")
base = Path(__file__).parent.parent / "runs" / "m4_cpg_v12"
for s in (0, 1, 2):
    mp = base / f"seed{s}" / "s3b_arc30" / "model_best.zip"
    if mp.exists():
        report(f"seed{s}", *analyze(mp))
    else:
        print(f"  seed{s}: 모델 없음")
