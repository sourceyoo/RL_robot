"""m4_cpg_v11 reach_gate(align 게이팅) floor 적절성 probe (read-only, 학습 trigger 아님).

질문: reach_gate = ALIGN_FLOOR + (1-FLOOR)·clip(align_avg) 가 게걸음/선회를 실제로 가르나?
→ v10 s3b 정책(게이팅 없이 학습)의 도달 ep 를 선회비율로 게걸음/선회 분류,
  도달직전 3-step align_avg 분포가 분리되는지 + reach 값 차이가 인센티브로 충분한지 측정.
분리 OK → floor 현행 0.4 유효. 분리 약함 → 게걸음도 도달직전 align 높음(회피) → window↑/신호 변경 재고.
사용: python3 diagnostics/reach_gate_probe.py
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv

PI = np.pi
ACTION_HISTORY_N = 20
CYCLE_STEPS = 3
N_EP = 50
ALIGN_FLOOR = 0.4
S3B_KW = dict(episode_seconds=60.0, success_radius=0.08, action_history_n=ACTION_HISTORY_N,
              target_theta_offset_range=(PI / 6, PI / 6))


def reach_gate(align_avg):
    return ALIGN_FLOOR + (1.0 - ALIGN_FLOOR) * float(np.clip(align_avg, 0.0, 1.0))


WINDOW_LATE = 12  # 도달직전 ~1s (12 step) — 도달 순간 노이즈 희석


def analyze(model_path):
    model = SAC.load(str(model_path), device="cpu")
    env = FishSwimEnv(**S3B_KW)
    rows = []  # (turn_ratio, ep_mean_align, late12_align)
    for ep in range(N_EP):
        obs, _ = env.reset(seed=4000 + ep)
        yaw0 = float(env.data.qpos[2])
        tgt = env._target_pos()[:2]
        offset = abs(np.arctan2(tgt[1], -tgt[0]))
        aligns = []
        term = trunc = False
        reached = False
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            aligns.append(float(info.get("align", 0.0)))
            reached = bool(info.get("reached", False))
            if reached:
                break
        if reached and offset > 1e-3:
            yaw_final = float(env.data.qpos[2])
            ep_mean = float(np.mean(aligns))
            late = float(np.mean(aligns[-WINDOW_LATE:]))  # 도달직전 12 step
            rows.append((abs(yaw_final - yaw0) / offset, ep_mean, late))
    env.close()
    return rows


def report(tag, rows):
    if not rows:
        print(f"  {tag}: 도달 ep 없음"); return
    r = np.array([x[0] for x in rows])
    # 게걸음/선회 그룹: 선회비율 하위/상위 (임계 0.4)
    crab_mask = r < 0.4
    turn_mask = r >= 0.4
    print(f"  {tag}: 도달 {len(rows)}ep  게걸음(<0.4)={crab_mask.sum()}  선회(≥0.4)={turn_mask.sum()}")
    for col, nm in [(1, "ep평균 align"), (2, "도달직전12 align")]:
        v = np.array([x[col] for x in rows])
        ac = float(np.median(v[crab_mask])) if crab_mask.any() else None
        at = float(np.median(v[turn_mask])) if turn_mask.any() else None
        s = f"    [{nm}] "
        if ac is not None:
            s += f"게걸음 median={ac:+.3f}→reach={10*reach_gate(ac):.2f}  "
        if at is not None:
            s += f"선회 median={at:+.3f}→reach={10*reach_gate(at):.2f}  "
        if ac is not None and at is not None:
            s += f">>> 차={10*(reach_gate(at)-reach_gate(ac)):+.2f}"
        print(s)


print(f"[reach_gate] v10 s3b 도달직전 3-step align_avg, 선회비율로 게걸음/선회 분류. FLOOR={ALIGN_FLOOR}, N_EP={N_EP}")
base = Path(__file__).parent.parent / "runs" / "m4_cpg_v10"
for s in (0, 1, 2):
    mp = base / f"seed{s}" / "s3b_arc30" / "model_best.zip"
    if mp.exists():
        report(f"seed{s}", analyze(mp))
    else:
        print(f"  seed{s}: 모델 없음")
