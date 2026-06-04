"""m4_cpg v9 vs v10 게걸음 정량 비교 (read-only, 학습 trigger 아님).

사용자 요구: P=10(v10)이 게걸음을 줄였나 + cos_motion 회피했나.
지표:
  - 선회비율 turn_ratio = |yaw_final - yaw0| / |target offset|.  1=순수선회, 0=게걸음(머리 안 돎).
  - cos_motion 분포: 게걸음이 cos_motion 을 높여 P 게이트를 회피했는지 (v9 게걸음 0.917).
도달(reached) ep 만 선회비율 집계 (도달 못한 ep는 게걸음 판정 무의미).
사용: python3 diagnostics/crab_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv

PI = np.pi
ACTION_HISTORY_N = 20
N_EP = 40
S3B_KW = dict(episode_seconds=60.0, success_radius=0.08, action_history_n=ACTION_HISTORY_N,
              target_theta_offset_range=(PI / 6, PI / 6))  # ±30° 고정


def analyze(model_path):
    model = SAC.load(str(model_path), device="cpu")
    env = FishSwimEnv(**S3B_KW)
    turn_ratios, cos_reached = [], []
    for ep in range(N_EP):
        obs, _ = env.reset(seed=3000 + ep)
        yaw0 = float(env.data.qpos[2])
        # target offset (머리 −x 에서 돌아야 할 각)
        tgt = env._target_pos()[:2]
        tgt_ang = np.arctan2(tgt[1], -tgt[0])  # −x 기준 편차 (head_dir=[-cos,sin] 좌표계)
        offset = abs(tgt_ang)
        prev_xy = env._torso_pos()[:2].copy()
        cos_ep = []
        term = trunc = False
        reached = False
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            xy = env._torso_pos()[:2]
            yaw = float(env.data.qpos[2])
            hd = np.array([-np.cos(yaw), np.sin(yaw)])
            d = xy - prev_xy
            dn = float(np.linalg.norm(d))
            if dn > 1e-4:
                cos_ep.append(float(np.dot(d, hd)) / dn)
            prev_xy = xy.copy()
            reached = bool(info.get("reached", False))
        yaw_final = float(env.data.qpos[2])
        if reached and offset > 1e-3:
            turn_ratios.append(abs(yaw_final - yaw0) / offset)
            cos_reached.extend(cos_ep)
    env.close()
    return np.array(turn_ratios), np.array(cos_reached)


def report(tag, ratios, cosm):
    if len(ratios) == 0:
        print(f"  {tag}: 도달 ep 없음"); return
    rp = np.percentile(ratios, [25, 50, 75])
    cp = np.percentile(cosm, [10, 50, 90]) if len(cosm) else [0, 0, 0]
    print(f"  {tag}:  선회비율 n={len(ratios)} mean={ratios.mean():.2f} "
          f"(25/50/75 = {rp[0]:.2f}/{rp[1]:.2f}/{rp[2]:.2f})  "
          f"|  cos_motion 10/50/90 = {cp[0]:.3f}/{cp[1]:.3f}/{cp[2]:.3f}")


print(f"[crab] s3b(±30°) 선회비율(머리돈각/offset) + cos_motion. N_EP={N_EP}, 도달 ep만.")
print("  선회비율 1=순수선회, 0=게걸음. cos_motion: 게걸음이 P게이트 회피하려 높였는지(v9 0.917).")
base = Path(__file__).parent.parent / "runs"
for tag, card in [("v9 ", "m4_cpg_v9"), ("v10", "m4_cpg_v10")]:
    print(f"\n{tag} ({card}):")
    for s in (0, 1, 2):
        mp = base / card / f"seed{s}" / "s3b_arc30" / "model_best.zip"
        if not mp.exists():
            print(f"  seed{s}: 모델 없음 ({mp})"); continue
        r, c = analyze(mp)
        report(f"seed{s}", r, c)
