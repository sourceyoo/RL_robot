"""m4_cpg_v10 PROGRESS_P↑ 안전 측정용 probe (read-only, 학습 trigger 아님).

critic watch-point: P=10 으로 키우면 정상 직진(s1)·완만 선회(s3a)에서 head wag 로
cos_motion 이 1보다 낮을 때 progress 가 과감쇠(추진 학습 저하)될 위험.
v9 정책 rollout 에서 step별 cos_motion 분포를 측정해 s1/s3a 가 안전한지(≈0.95↑) 확인.

cos_motion = (step 변위 · 머리방향) / |step 변위|.  fish_env reward 계산과 동일 정의.
사용: python3 diagnostics/cos_motion_probe.py
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv

PI = np.pi
ACTION_HISTORY_N = 20
N_EP = 30
RUNS = Path(__file__).parent.parent / "runs" / "m4_cpg_v9" / "seed0"

CASES = [
    {"name": "s1_forward (직진)", "model": RUNS / "s1_forward" / "model_best.zip",
     "kwargs": dict(episode_seconds=20.0, success_radius=0.08, action_history_n=ACTION_HISTORY_N,
                    target_theta_range=(PI, PI))},
    {"name": "s3a_arc15 (완만 선회)", "model": RUNS / "s3a_arc15" / "model_best.zip",
     "kwargs": dict(episode_seconds=60.0, success_radius=0.08, action_history_n=ACTION_HISTORY_N,
                    target_theta_offset_range=(PI / 12, PI / 12))},
]


def collect(model_path, kwargs):
    """각 step 의 cos_motion (delta 정의) 수집. delta_norm 너무 작은 step(정지)은 제외."""
    model = SAC.load(str(model_path), device="cpu")
    env = FishSwimEnv(**kwargs)
    cos_all = []
    for ep in range(N_EP):
        obs, _ = env.reset(seed=2000 + ep)
        prev_xy = env._torso_pos()[:2].copy()
        term = trunc = False
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            xy = env._torso_pos()[:2]
            yaw = float(env.data.qpos[2])
            head_dir = np.array([-np.cos(yaw), np.sin(yaw)])
            delta = xy - prev_xy
            dn = float(np.linalg.norm(delta))
            if dn > 1e-4:  # 실제 이동한 step 만 (정지 step 의 cos 는 무의미)
                cos_all.append(float(np.dot(delta, head_dir)) / dn)
            prev_xy = xy.copy()
    env.close()
    return np.array(cos_all)


def report(name, sig):
    pct = np.percentile(sig, [5, 10, 25, 50, 75])
    print(f"  {name:24s} n={len(sig):5d}  mean={sig.mean():.4f}  "
          f"pct 5/10/25/50/75 = {pct[0]:.4f}/{pct[1]:.4f}/{pct[2]:.4f}/{pct[3]:.4f}/{pct[4]:.4f}")
    print(f"  {'':24s} gate^P (P=10): mean={sig.mean()**10 if sig.mean()>0 else 0:.3f}  "
          f"10th={max(pct[1],0)**10:.3f}  (P=5 비교: 10th={max(pct[1],0)**5:.3f})")


print(f"[probe] v9 정책 cos_motion 분포 (N_EP={N_EP}). 하위 percentile 이 P=10 으로 얼마나 깎이는지가 핵심.")
for c in CASES:
    sig = collect(c["model"], c["kwargs"])
    print(f"\n{c['name']}")
    report("cos_motion", sig)
print("\n해석: s1 10th percentile 의 ^10 값이 0.5 이상이면 직진 추진 보상 유지 안전.")
print("      0.3 이하로 떨어지면 head wag 로 정상 직진까지 과감쇠 → P 재고(7~8) 또는 직진 영향 인지하고 진행.")
