"""m4_cpg_v8 YAW_GATE 결정용 probe (read-only, 학습 trigger 아님).

v7 best 모델로 cycle-평균 yaw_rate · yaw_err_sign 분포를 측정해 게이트 임계값을 잡는다.
- s1_forward (직진 baseline): 면제되면 안 되는 쪽. 게걸음도 머리 고정(yaw_avg≈0)이라 직진과 유사 → 게걸음 proxy.
- s3b_arc30 (정상 선회 baseline): 면제돼야 하는 쪽 → 신호 양수.
YAW_GATE = 직진 95th percentile 과 선회 상위 분포 사이의 골.

신호 정의는 fish_env v8 게이트와 동일: mean(yaw_rate, 3 step) · yaw_err_sign.
사용: python3 diagnostics/yaw_gate_probe.py
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv

PI = np.pi
CYCLE_STEPS = 3
IDX_YAW = 2
ACTION_HISTORY_N = 20
N_EP = 30

RUNS = Path(__file__).parent.parent / "runs" / "m4_cpg_v7" / "seed0"

CASES = [
    {"name": "s1_forward (직진/게걸음 proxy)", "model": RUNS / "s1_forward" / "model_best.zip",
     "kwargs": dict(episode_seconds=20.0, success_radius=0.08, action_history_n=ACTION_HISTORY_N,
                    target_theta_range=(PI, PI))},
    {"name": "s3b_arc30 (정상 선회)", "model": RUNS / "s3b_arc30" / "model_best.zip",
     "kwargs": dict(episode_seconds=60.0, success_radius=0.08, action_history_n=ACTION_HISTORY_N,
                    target_theta_offset_range=(PI / 12, PI / 6))},
]


def collect(model_path, kwargs):
    """각 step 에서 align-delta 두 버전 수집: d1=1-step, d3=3-step 차분(head wag 상쇄)."""
    model = SAC.load(str(model_path), device="cpu")
    env = FishSwimEnv(**kwargs)
    d1_all, d3_all = [], []
    for ep in range(N_EP):
        obs, _ = env.reset(seed=1000 + ep)
        abuf = np.zeros(CYCLE_STEPS + 1)  # align history (현재 + 3 step 전까지)
        _, _, _, _, info0 = env.step(model.predict(obs, deterministic=True)[0])
        abuf[:] = float(info0.get("align", 0.0))
        obs = env._get_obs()
        term = trunc = False
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            a = float(info.get("align", 0.0))
            abuf[:-1] = abuf[1:]
            abuf[-1] = a
            d1_all.append(abuf[-1] - abuf[-2])        # 1-step delta
            d3_all.append(abuf[-1] - abuf[0])         # 3-step delta (head wag 상쇄)
    env.close()
    return np.array(d1_all), np.array(d3_all)


def report(name, sig):
    pct = np.percentile(sig, [50, 75, 90, 95, 99])
    print(f"  {name:5s}  mean={sig.mean():+.5f} std={sig.std():.5f}  "
          f"pct50/75/90/95/99={pct[0]:+.5f}/{pct[1]:+.5f}/{pct[2]:+.5f}/{pct[3]:+.5f}/{pct[4]:+.5f}  "
          f">0={(sig > 0).mean() * 100:.0f}%")
    return pct


print(f"[probe] N_EP={N_EP}, signal = align-delta (d1=1step, d3=3step). 선회=align 증가>0, 직진/게걸음=0근처 기대")
results = {}
for c in CASES:
    d1, d3 = collect(c["model"], c["kwargs"])
    results[c["name"]] = {"d1": d1, "d3": d3}
    print(f"\n{c['name']}  (n={len(d1)})")
    report("d1", d1)
    report("d3", d3)

s_name, t_name = CASES[0]["name"], CASES[1]["name"]
for key in ("d1", "d3"):
    straight = results[s_name][key]
    turn = results[t_name][key]
    gate_lo = float(np.percentile(straight, 95))
    turn_75 = float(np.percentile(turn, 75))
    turn_90 = float(np.percentile(turn, 90))
    print(f"\n[{key}] 직진 95th={gate_lo:+.5f}  선회 75th/90th={turn_75:+.5f}/{turn_90:+.5f}")
    if turn_90 > gate_lo > 0:
        print(f"  → 분리 OK. GATE ≈ {(gate_lo + turn_90) / 2:+.5f} 권장 (직진 95th ~ 선회 90th 사이)")
    else:
        print(f"  → 분리 미흡.")
