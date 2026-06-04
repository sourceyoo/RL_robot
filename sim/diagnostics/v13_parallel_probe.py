"""m4_cpg_v13 parallel mode 검증 (read-only, 학습 trigger 아님).

plan 분석 항목:
 ① 머리 회전 각도(도): v12 seed0 +0.3° → v13 ↑? (목표 s3a 15°, s3b 30°). 실제 도수로 판단.
 ② 계단식 지표 = 회전 중 전진속도 + 회전집중도: v12 seed2 -0.70 → v13 양수 유지?
 ③ sideslip: v12 29~51° → ↓?
 ④ 게걸음 reach 안주: 도달 ep 중 turn_ratio<0.2(머리 거의 안 돎) 비율. 높으면 FLOOR이 게걸음 떠받침.

staircase_probe.py 와 동일 로직 + turn_ratio(env._yaw0·_turn_sign·_target_offset 직접 계산) 추가.
사용: python3 diagnostics/v13_parallel_probe.py
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

STAGES = {
    "s3a_arc15": (PI / 12, PI / 12),  # ±15°
    "s3b_arc30": (PI / 6, PI / 6),    # ±30°
}


def analyze(model_path, offset_range):
    model = SAC.load(str(model_path), device="cpu")
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08,
                      action_history_n=ACTION_HISTORY_N,
                      target_theta_offset_range=offset_range)
    head_turn_deg, slip_deg, turn_conc, vfwd_min_frac, final_tr = [], [], [], [], []
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
        # 최종 turn_ratio (env 와 동일 식)
        tr = np.clip((yaws[-1] - env._yaw0) * env._turn_sign / env._target_offset, 0.0, 1.0) \
            if env._target_offset > 1e-3 else 1.0
        final_tr.append(float(tr))
        spd = np.hypot(vfwd, vlat)
        mv = spd > 1e-3
        if mv.any():
            slip = np.degrees(np.abs(np.arctan2(vlat[mv], vfwd[mv])))
            slip_deg.append(float(np.mean(slip)))
        dyaw = np.abs(np.diff(yaws))
        if dyaw.sum() > 1e-4:
            k = max(1, int(0.2 * len(dyaw)))
            turn_conc.append(float(np.sort(dyaw)[-k:].sum() / dyaw.sum()))
        if np.median(vfwd) > 1e-4:
            vfwd_min_frac.append(float(np.percentile(vfwd, 5) / np.median(vfwd)))
    env.close()
    return dict(ht=np.array(head_turn_deg), sl=np.array(slip_deg),
                tc=np.array(turn_conc), vf=np.array(vfwd_min_frac),
                tr=np.array(final_tr))


def report(tag, target_deg, r):
    ht = r["ht"]
    if len(ht) == 0:
        print(f"  {tag}: 도달 ep 없음"); return
    crab = float(np.mean(r["tr"] < 0.2))  # ④ 게걸음 안주 비율
    print(f"  {tag}: 도달 {len(ht)}/{N_EP}ep")
    print(f"    ① 머리회전(도)   mean={ht.mean():+6.1f}  |절대|median={np.median(np.abs(ht)):.1f}  (목표 ±{target_deg})")
    print(f"    ② 회전중전진비   mean={r['vf'].mean():+.2f}  (멈춤=음수/0, parallel=0.3~1)   회전집중도 mean={r['tc'].mean():.2f} (0.2=parallel,→1=계단)")
    print(f"    ③ sideslip(도)   mean={r['sl'].mean():6.1f}  median={np.median(r['sl']):.1f}   (0=정렬,90=게걸음)")
    print(f"    ④ 최종turn_ratio mean={r['tr'].mean():.2f}   게걸음안주(tr<0.2) 비율={crab*100:.0f}%")


print(f"[v13 parallel probe] N_EP={N_EP}, 도달 ep만. 머리 돌며 전진(parallel)인지 검증.")
CARD = sys.argv[1] if len(sys.argv) > 1 else "m4_cpg_v13"
print(f"[card={CARD}]")
base = Path(__file__).parent.parent / "runs" / CARD / "seed0"
for stage, off in STAGES.items():
    mp = base / stage / "model_best.zip"
    if not mp.exists():
        mp = base / stage / "model.zip"
    if mp.exists():
        target_deg = 15 if "15" in stage else 30
        report(f"{stage} ({mp.name})", target_deg, analyze(mp, off))
    else:
        print(f"  {stage}: 모델 없음")
