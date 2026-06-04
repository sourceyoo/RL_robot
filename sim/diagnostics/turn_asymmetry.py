"""좌/우 선회 비대칭 진단 (read-only, 학습 trigger 아님).

사용자 관찰: "선회는 잘되는데 좌회전이 안된다". probe(turn_sign 정규화)가 가린 방향 비대칭 확인.
target = π ± offset (sign ±50% 랜덤). _turn_sign 으로 좌/우 그룹 분리해 도달률·머리회전·최종거리 비교.
실제 curriculum range 사용: s3a [9.2°,15°], s3b [15°,30°].
사용: python3 diagnostics/turn_asymmetry.py [card]
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv

PI = np.pi
DT = 0.084
N_EP = 100
STAGES = {"s3a_arc15": (PI * 9.2 / 180.0, PI / 12), "s3b_arc30": (PI / 12, PI / 6)}


def analyze(model_path, off):
    model = SAC.load(str(model_path), device="cpu")
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                      target_theta_offset_range=off)
    # 그룹: turn_sign = +1 / -1
    g = {+1: dict(n=0, reach=0, head=[], fdist=[]), -1: dict(n=0, reach=0, head=[], fdist=[])}
    for ep in range(N_EP):
        obs, _ = env.reset(seed=4000 + ep)
        ts = int(np.sign(env._turn_sign)) if env._turn_sign != 0 else 1
        yaw0 = float(env.data.qpos[2])
        term = trunc = reached = False
        last_yaw = yaw0
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            last_yaw = float(env.data.qpos[2])
            reached = bool(info.get("reached", False))
            if reached:
                break
        d = g[ts]
        d["n"] += 1
        d["reach"] += int(reached)
        # 머리회전을 turn_sign 방향으로 부호화(+ = target쪽으로 돈 정도)
        d["head"].append(np.degrees((last_yaw - yaw0) * ts))
        d["fdist"].append(env._distance_to_target())
    env.close()
    return g


def report(tag, g):
    print(f"  {tag}:")
    for ts, name in [(+1, "sign=+1 (한쪽)"), (-1, "sign=-1 (반대쪽)")]:
        d = g[ts]
        if d["n"] == 0:
            print(f"    {name}: ep 없음"); continue
        rr = 100.0 * d["reach"] / d["n"]
        head = np.array(d["head"]); fd = np.array(d["fdist"])
        print(f"    {name}: {d['n']}ep  도달 {rr:4.0f}%  "
              f"머리회전(target쪽+) mean {head.mean():+6.1f}°  최종거리 mean {fd.mean():.3f} (성공<0.08)")


print(f"[turn asymmetry] N_EP={N_EP}, deterministic. 좌/우 선회 도달률·머리회전 분리.")
CARD = sys.argv[1] if len(sys.argv) > 1 else "m4_cpg_v16"
SEED = sys.argv[2] if len(sys.argv) > 2 else "seed0"
print(f"[card={CARD}, {SEED}]")
base = Path(__file__).parent.parent / "runs" / CARD / SEED
for stage, off in STAGES.items():
    mp = base / stage / "model_best.zip"
    if not mp.exists():
        mp = base / stage / "model.zip"
    report(stage, analyze(mp, off)) if mp.exists() else print(f"  {stage}: 모델 없음")
